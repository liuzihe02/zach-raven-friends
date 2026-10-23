(* MLP with Vega/Rune *)
(* no open Nx for clarity of namespace*)

(* define dims *)
let d_in = 2
let d_hidden = 4
let d_out = 2
let bs = 128

(* big nn module, struct is just declaring module body *)
module Mlp = struct
  (*reusable params module*)
  module Params = struct
    type t =
      { w1 : Nx.float32_t
      ; w2 : Nx.float32_t
      ; b1 : Nx.float32_t
      ; b2 : Nx.float32_t
      }
    [@@deriving ptree]
    (* [@@deriving ptree] adds [walk] and [ptree : t Nx.Ptree.t] beside [t];
       P.ptree is the Raven structure for rune vega etc *)
  end

  let init d_in d_hidden d_out =
    Nx.Infix.(
      (* He init, for ReLU *)
      let scale fan_in = Float.sqrt (2.0 /. Float.of_int fan_in) in
      let w1 = Nx.randn Nx.float32 [| d_in; d_hidden |] *$ scale d_in
      and w2 = Nx.randn Nx.float32 [| d_hidden; d_out |] *$ scale d_hidden
      and b1 = Nx.randn Nx.float32 [| d_hidden |]
      and b2 = Nx.randn Nx.float32 [| d_out |] in
      Params.{ w1; w2; b1; b2 })

  let forward x (p : Params.t) =
    Nx.Infix.(
      let a = Kaun.Fn.relu ((x *@ p.w1) + p.b1) in
      let out = (a *@ p.w2) + p.b2 in
      out)
end

(* fixed value *)
let params_teacher = Mlp.init d_in d_hidden d_out

(* create training data
   CAREFUL this is fixed global data, will be compiled
   annoying for jit/minibatches later *)
let data =
  let x = Nx.randn Nx.float32 [| bs; d_in |] in
  let y = Mlp.forward x params_teacher in
  x, y

(* Train the student, with Rune for the gradients and Vega's Adam for the update. *)
let lr = Vega.lr 0.001

(* initial student params *)
let params_student = Mlp.init d_in d_hidden d_out

(* -- Optimiser state --
   adam_init takes the structure (ptree) and the actual params (tensors)
   Vega is generic over any params type and OCaml erases types at
   runtime, so the structure has to be passed in... (??confused??) *)

(* point to our custom structure **)
let ptree = Mlp.Params.ptree
let st = Vega.adam_init ptree params_student

(* -- Loss --
   accesses global data object *)
let loss params =
  let x, y = data in
  Kaun.Loss.mse (Mlp.forward x params) y

(* -- Step --
   step : params -> adam_state -> params * adam state * loss *)
let step params st =
  (* value_and_grad :
     'p Ptree.t ->
     ('p -> ('c, 'd) t) ->
     'p ->
     ('c, 'd) t * 'p
     --- why is this impossible?? to read??
     args: structure, loss fn (params -> scalar), params
     returns: (loss value, grads) *)
  let l, grads = Rune.value_and_grad ptree loss params in
  let params, st = Vega.adam_step ptree ~lr st ~params ~grads in
  params, st, l

(* -- Training loop --
   train : int -> Mlp.P.t -> Mlp.P.t Vega.adam_state -> Mlp.P.t
   Recursive, but carrying params AND Adam state *)
let steps = 10000

let rec train ~i params st =
  if i = steps
  then params
  else (
    let params, st, l = step params st in
    if i mod 1000 = 0 then Stdio.printf "%d %f\n" i (Nx.item [] l);
    (* confusing syntax *)
    train ~i:(i + 1) params st)

(* -- Main (let () = ...) --
   run train 0 params_student st, print the final loss. *)
let _ = train ~i:0 params_student st
