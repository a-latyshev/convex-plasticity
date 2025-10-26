# Convex-plasticity v2.0

## GNU Parallel

```shell
apt update
apt install parallel locales
```

## CVXPYGen

Requirements:

Auto-installations:
 - Julia

To use CVXPYGEN and CLARABEL, LOG:
```shell
apt install cargo
cargo install cbindgen --locked
export PATH="$PATH:/root/.cargo/bin"
apt install rustup
rustup update nightly
```

For cvxpylayers:

```shell
pip install jax==0.5.3
```
