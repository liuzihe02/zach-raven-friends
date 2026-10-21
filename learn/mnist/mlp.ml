open Base
open Nx

let print s = Stdio.print_endline (Sexp.to_string_hum s)

(* MLP teacher-student setup *)

module MLP = struct
  module P = struct
    type 'a t =
      { weights : 'a
      ; biases : 'a
      }
    [@@deriving ptree]
  end

  let init n_in n_out =
    let open Infix in
    let weights =
      let z = Float.(sqrt (of_int n_in)) in
      randn float32 [| n_in; n_out |] /$ z
    in
    let biases = randn float32 [| n_out |] in
    P.{ weights; biases }

  (* x : (batch_size, n_in) → input tensor
     return (bs, n_out) output tensor *)
  let eval params x =
    let open Infix in
    tanh ((x *@ params.P.weights) + params.biases)
end

let n_in = 16
let n_out = 8
let bs = 128
let teacher = MLP.init n_in n_out
let student = MLP.init n_in n_out

(* use the teacher to create some data to train the student on *)
let data =
  let x = randn float32 [| bs; n_in |] in
  let y = MLP.eval teacher x in
  x, y

(* construct a training loop with Vega's adam optimiser to optimise
   the student parameters such that it captures the (x,y) data pairs obtained above from the teacher *)
