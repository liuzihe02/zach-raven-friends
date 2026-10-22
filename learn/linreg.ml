(* Linear regression trained by gradient descent, with hand-derived gradients *)

(* -- Data --
   Make a synthetic dataset
   - true_w : [d; 1], true_b : [1]   (pick d = 3, say)
   - x      : [n; d], random inputs         (n = 64, say)
   - y      : [n; 1] = x true_w + true_b
*)

(* define shapes *)
let d_in = 3
let d_out = 1
let n = 64

(* define data *)
let true_w = Nx.create Nx.float32 [| d_in; d_out |] [| 1.0; 2.0; 3.0 |]
let true_b = Nx.create Nx.float32 [| d_out |] [| 1.0 |]
let data_x = Nx.randn Nx.float32 [| n; d_in |]
let data_y = Nx.Infix.((data_x *@ true_w) + true_b)

(* -- Parameters --
   pure record type { w; b }. *)
type params =
  { w : Nx.float32_t
  ; b : Nx.float32_t
  }

(* -- Loss --
   A function: params -> scalar tensor. Predict, subtract y, square, average. *)
let residual x y (params : params) =
  let predict = Nx.Infix.((x *@ params.w) + params.b) in
  Nx.Infix.(predict - y)

let loss residual =
  let squares = Nx.square residual in
  Nx.mean squares

(* -- Gradients (by hand) --
   A function: params -> gradients, returning a params record { w; b } *)
(* let grad of any general data x y*)
let grad x y (params : params) =
  let r = residual x y params in
  let scale = 2.0 /. Float.of_int (Nx.dim 0 x) in
  (* take dim from x dont hardcode global*)
  (* custom grad functions for the different params *)
  let dw = Nx.Infix.(Nx.transpose x *@ r *$ scale) in
  let db = Nx.Infix.(Nx.sum ~axes:[ 0 ] r *$ scale) in
  { w = dw; b = db }

(* -- Initial params --
   A params record: zeros. *)
let params_init = { w = Nx.zeros_like true_w; b = Nx.zeros_like true_b }

(* Hyperparameters: lr (a float) and n_steps (an int). *)
let lr = 0.001
let steps = 10000

(* step : params -> params
   One gradient-descent update*)
let step x y (params : params) =
  let g = grad x y params in
  (* you cant apply tensor operations on a ptree, only tensors*)
  { w = Nx.Infix.(params.w - (g.w *$ lr)); b = Nx.Infix.(params.b - (g.b *$ lr)) }

(* train : int -> params -> params *)
let rec train x y i params =
  if i = steps
  then params (* base case done: return the final state *)
  else train x y (i + 1) (step x y params)

(* main *)
let params_final = train data_x data_y 0 params_init

(*check *)
let () =
  Nx.print params_final.w;
  Nx.print params_final.b
