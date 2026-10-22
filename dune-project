(lang dune 3.24)

(package
 (name learn) ; package for the learn/ exercises
 (allow_empty) ; this means package installs nothing
 (depends base stdio ppx_sexp_message)) ; only for opam packages, leave out raven, sofo-raven

; cmdargs is the lab's own package (needed by sofo-raven), not on opam, so fetch it from GitHub
(pin
 (url "git+https://github.com/hennequin-lab/cmdargs.git")
 (package
  (name cmdargs)))