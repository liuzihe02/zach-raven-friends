# zach-raven-friends

Dune workspace for working with [sofo-raven](https://github.com/hennequin-lab/sofo-raven).

## Setup

Clone both dependencies into the workspace root (they are gitignored):

```bash
git clone https://github.com/raven-ml/raven.git
git clone https://github.com/hennequin-lab/sofo-raven.git
```

commit pins here:

- raven: `29de40a34`
- sofo-raven: `f0e3ecc`

Then build:

```bash
dune pkg lock   # only needed when depends change
dune build
```