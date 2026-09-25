# Arbitrary Fractional-Order Photonic Differentiator (MRR)

Project for the **Photonic Computing** course at Politecnico di Milano.  
Simulation framework and tolerance analysis for all-optical fractional-order differentiators based on microring resonator (MRR) cascades.

---

## Build

From the root directory, create the `build` folder and compile:

```bash
mkdir build && cd build
cmake ..
make
```

## Run

Run the executable passing the target differentiation order -n:

```Bash
./MRR -n 0.54
./MRR -n 1.44
./MRR -n 2.10
```

Or a custom order with specific Gaussian pulse width T0 in picoseconds, for example

```bash
./MRR -n 0.75 -t 4.5
```

Options:

- `-n, --order <val>` : Differentiation order n>0 (required).
- `-t, --pulse <ps>` : Input pulse half-width T0 in picoseconds (optional; if omitted, automatically matched to the filter band).
- `-h, --help` : Show CLI help.
