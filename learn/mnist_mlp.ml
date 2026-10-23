(* MNIST classification: MLP with Rune + Vega Adam, training step compiled with
   Rune.jit *)

let d_in = 784 (* 28 x 28 pixels, flattened *)
let d_hidden = 128
let d_out = 10 (* one logit per digit class *)
let bs = 128 (* batch size here refers to minibatch size*)
let epochs = 5

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
let lr = Vega.lr 0.01
let ptree = Mlp.Params.ptree
let params_init = Mlp.init d_in d_hidden d_out
let opt_st_init = Vega.adam_init ptree params_init
let step_st_init = params_init, opt_st_init

(* -- Loss --
   loss : x [bs; 784] -> y [bs] int64 labels -> params -> scalar
   Softmax cross-entropy between the logits and the integer labels: see kaun/lib/loss.mli
   softmax_cross_entropy_sparse is for int labels *)
let loss x y params = Kaun.Loss.softmax_cross_entropy_sparse (Mlp.forward x params) y

(* -- Step (same as mlp_jit) -- *)
let step batch step_st =
  let x, y = batch in
  let params, opt_st = step_st in
  let loss_params = loss x y in
  let l, grads = Rune.value_and_grad ptree loss_params params in
  let params, opt_st = Vega.adam_step ptree ~lr opt_st ~params ~grads in
  (* just for clarity*)
  let step_st = params, opt_st in
  l, step_st

let step_st_sign = Nx.Ptree.pair ptree (Vega.adam_ptree ptree)

let step_sig =
  Nx.Ptree.(
    pair tensor tensor @-> consumes step_st_sign @@ returns (pair tensor step_st_sign))

let step_jit = Rune.jit step_sig step

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
  Seq.fold_left (fun step_st batch -> snd (step_jit batch step_st)) step_st batches

(* -- Epochs --
   train_epochs : int -> state -> state, recursive like mlp_jit's train *)
let rec train_epochs ~i step_st =
  if i = epochs
  then step_st
  else (
    let step_st = train_epoch step_st in
    let test_loss = loss test_x test_y (fst step_st) in
    Stdio.printf "epoch %d  test loss %.4f\n%!" i (Nx.item [] test_loss);
    train_epochs ~i:(i + 1) step_st)

(* -- Main: train, then test accuracy -- *)
let () =
  let params, _ = train_epochs ~i:0 step_st_init in
  let logits = Mlp.forward test_x params in
  Printf.printf "test accuracy: %.4f\n" (Kaun.Metric.accuracy logits test_y)
