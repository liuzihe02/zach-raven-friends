(* Teacher-student MLP with Rune + Vega Adam, training step compiled with Rune.jit *)
(* no open Nx for clarity of namespace*)

let d_in = 2
let d_hidden = 4
let d_out = 2
let bs = 128

module Mlp = struct
  module Params = struct
    type t =
      { w1 : Nx.float32_t
      ; w2 : Nx.float32_t
      ; b1 : Nx.float32_t
      ; b2 : Nx.float32_t
      }
    [@@deriving ptree]
    (* also generates [ptree : t Nx.Ptree.t], the structure Rune and Vega walk *)
  end

  (* He init, for ReLU *)
  let init d_in d_hidden d_out =
    Nx.Infix.(
      let scale fan_in = Float.sqrt (2.0 /. Float.of_int fan_in) in
      let w1 = Nx.randn Nx.float32 [| d_in; d_hidden |] *$ scale d_in
      and w2 = Nx.randn Nx.float32 [| d_hidden; d_out |] *$ scale d_hidden
      and b1 = Nx.randn Nx.float32 [| d_hidden |]
      and b2 = Nx.randn Nx.float32 [| d_out |] in
      Params.{ w1; w2; b1; b2 })

  let forward x (p : Params.t) =
    Nx.Infix.(
      let a = Kaun.Fn.relu ((x *@ p.w1) + p.b1) in
      (a *@ p.w2) + p.b2)
end

let params_teacher = Mlp.init d_in d_hidden d_out
let params_student = Mlp.init d_in d_hidden d_out

(* training data, passed to loss and step as arguments (not captured) *)
let x = Nx.randn Nx.float32 [| bs; d_in |]
let y = Mlp.forward x params_teacher
let lr = Vega.lr 0.001
let ptree = Mlp.Params.ptree

(* loss of the params on data x, y *)
let loss x y params = Kaun.Loss.mse (Mlp.forward x params) y

(* data are arguments, so the same step runs on any minibatch
   see below on why we have (param, st) tuple instead of flat*)

let step x y step_st =
  let params, opt_st = step_st in
  (* make loss(params) using partial functions *)
  let loss_params = loss x y in
  let l, grads = Rune.value_and_grad ptree loss_params params in
  let params, opt_st = Vega.adam_step ptree ~lr opt_st ~params ~grads in
  l, (params, opt_st)

(* TODO: define step_jit *)

(* -- Compiled step --

   step_jit : same as step, compiled once by Rune.jit and replayed after.

   Rune.jit takes a signature (Nx.Ptree's @->, returns, pair) listing the
   structure of each argument and of the result, then the function.

   - params: ptree; Adam state: Vega.adam_ptree ptree
   - there is no triple combinator, so return nested pairs
*)

(* note that our jitted func singature must return a type from Nx.Ptree.t
   this only has
   - unit
   - pair (allows mixing types)
   - list (must be same types)

   hence if we wanted to return a triplet mixing types,
   we have to construct them in terms of pairs...**)
let step_st_sign = Nx.Ptree.pair ptree (Vega.adam_ptree ptree)

let step_sig =
  Nx.Ptree.(
    tensor @-> tensor @-> consumes step_st_sign @@ returns (pair tensor step_st_sign))

let step_jit = Rune.jit step_sig step

(* train : int -> Mlp.Params.t * Mlp.Params.t Vega.adam_state -> Mlp.Params.t *)
let steps = 10000

(* Adam state: zero moments shaped like the params. Vega needs the structure
   (where the tensors are) and the params (the tensors themselves). *)
let opt_st = Vega.adam_init ptree params_student
let step_st = params_student, opt_st

let rec train ~i step_st =
  if i = steps
  then (
    let params, opt_st = step_st in
    params)
  else (
    let l, step_st = step_jit x y step_st in
    if i mod 1000 = 0 then Stdio.printf "%d %f\n" i (Nx.item [] l);
    train ~i:(i + 1) step_st)

let _ = train ~i:0 step_st
