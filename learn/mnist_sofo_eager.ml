(* MNIST classification: MLP trained with SOFO (sofo-raven) instead of Adam.
   Copied from mnist_mlp.ml. Eager: no Rune.jit. *)

let d_in = 784 (* 28 x 28 pixels, flattened *)
let d_hidden = 128
let d_out = 10 (* one logit per digit class *)
let bs = 128 (* batch size here refers to minibatch size*)
let epochs = 5
let k = 128 (* SOFO: number of random tangent directions per step *)
let print_every = 25 (* steps between loss prints *)

module Mlp = struct
  module Params = struct
    type t =
      { w1 : Nx.float32_t
      ; w2 : Nx.float32_t
      ; b1 : Nx.float32_t
      ; b2 : Nx.float32_t
      }
    [@@deriving ptree]
  end

  let init d_in d_hidden d_out =
    Nx.Infix.(
      let scale fan_in = Float.sqrt (2.0 /. Float.of_int fan_in) in
      let w1 = Nx.randn Nx.float32 [| d_in; d_hidden |] *$ scale d_in
      and w2 = Nx.randn Nx.float32 [| d_hidden; d_out |] *$ scale d_hidden
      and b1 = Nx.randn Nx.float32 [| d_hidden |]
      and b2 = Nx.randn Nx.float32 [| d_out |] in
      Params.{ w1; w2; b1; b2 })

  (* x : [bs; d_in] -> logits : [bs; d_out] *)
  let forward x (p : Params.t) =
    Nx.Infix.(
      let a = Kaun.Fn.relu ((x *@ p.w1) + p.b1) in
      (a *@ p.w2) + p.b2)
end

(* -- Data --
   Kaun_datasets.mnist () downloads once to ~/.cache/raven/datasets/mnist

   Images come as [n; 1; 28; 28] float32 in [0, 1],
   labels as [n] int64 class indices. *)
let train_x, train_y, test_x, test_y = Kaun_datasets.mnist ()

(* flatten [n; 1; 28; 28] images to [n; 784] for the MLP.
   -1 means "infer this axis from the total size", so it becomes n *)
let flatten x = Nx.reshape [| -1; d_in |] x
let train_x = flatten train_x
let test_x = flatten test_x
let lr = 1.0 (* SOFO: a plain float; 1.0 is the damped Newton step in the subspace *)
let damping : Sofo.Optim.damping = `Relative_from_top 1e-1 (* 1e-3 diverges at lr 1.0 *)
let ptree = Mlp.Params.ptree
let params_init = Mlp.init d_in d_hidden d_out

(* SOFO state: only the key the directions are drawn from, and a step counter *)
let opt_st_init = Sofo.Optim.init ~key:(Nx.Rng.key 0) ()
let step_st_init = params_init, opt_st_init

(* -- Loss --
   loss : x [bs; 784] -> y [bs] int64 labels -> params -> scalar
   Softmax cross-entropy between the logits and the integer labels: see kaun/lib/loss.mli
   softmax_cross_entropy_sparse is for int labels *)
let loss x y params = Kaun.Loss.softmax_cross_entropy_sparse (Mlp.forward x params) y

(* -- Marked loss --
   Kaun's sparse cross-entropy, marked with its exact curvature for the sketch.
   Sofo.softmax_ce does the same but fails inside a sketch (learn.md, Known
   issues): its curvature uses softmax of the tracked logits. Detaching them
   keeps the value and drops the derivative, which is exact here. *)
let softmax_ce logits labels =
  let l = Kaun.Loss.softmax_cross_entropy_sparse logits labels in
  let p = Nx.softmax (Rune.detach logits) in
  let rows = float_of_int (Nx.dim 0 logits) in
  Sofo.observe ~y:logits ~curv:(Sofo.Curv.softmax_ce ~scale:(1.0 /. rows) p) l;
  l

(* -- Step --
   Sofo.Optim.step *)
let step (x, y) (params, opt_st) =
  (* sofo's own version, fails inside the sketch (learn.md, Known issues):
     let loss_params p = Sofo.softmax_ce (Mlp.forward x p) y in *)
  let loss_params p = softmax_ce (Mlp.forward x p) y in
  let params, opt_st, sketch =
    Sofo.Optim.step ptree ~k ~lr ~damping opt_st ~loss:loss_params ~params
  in
  sketch.loss, (params, opt_st)

(* -- One epoch --

   One pass over the training data: a fresh shuffle, then fold step_jit over
   the minibatches.

   Seq.fold_left (fun accumulator elem -> new_accumlator) init_accum list_of_elems
   where new_acc computed from acc and element

   Seq.fold_left takes the accumulator first, then the elements
   the accumulator is the step_st (params, opt_st)

   fst returns first elem of a pair
   snd returns second element of a pair
   returns the accumulator step_st

   ~drop_last:true skips the smaller last batch (60000 = 468 x 128 + 96), so
   every batch has the same shape and step_jit never recompiles. *)
let train_epoch step_st =
  let batches =
    Kaun.Data.batches2 ~shuffle:true ~drop_last:true ~batch_size:bs (train_x, train_y)
  in
  (* fold_lefti also passes the step index i, to print every print_every steps *)
  Seq.fold_lefti
    (fun step_st i batch ->
       let l, step_st = step batch step_st in
       if i mod print_every = 0 then Stdio.printf "  step %d  loss %.4f\n%!" i (Nx.item [] l);
       step_st)
    step_st
    batches

(* -- Epochs --
   train_epochs : int -> state -> state, recursive like mlp_jit's train *)
let rec train_epochs ~i step_st =
  if i = epochs
  then step_st
  else (
    Stdio.printf "epoch %d\n%!" i;
    let step_st = train_epoch step_st in
    let test_loss = loss test_x test_y (fst step_st) in
    Stdio.printf "epoch %d done  test loss %.4f\n%!" i (Nx.item [] test_loss);
    train_epochs ~i:(i + 1) step_st)

(* -- Main: train, then test accuracy -- *)
let () =
  let params, _ = train_epochs ~i:0 step_st_init in
  let logits = Mlp.forward test_x params in
  Printf.printf "test accuracy: %.4f\n" (Kaun.Metric.accuracy logits test_y)
