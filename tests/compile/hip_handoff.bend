# Multiple offloads separated by CPU IO and array construction. The second
# marked call consumes data returned by the first, after a host update.
import Base

def fill(n: Nat, x: U32) -> Array<U32>:
  [x: U32 ^ n]

def changed(a: Array<U32>) -> Array<U32>:
  a[0] <- 11

def sum(a: Array<U32>) -> U32:
  match a:
    case ALeaf{x}:
      x
    case ANode{l, r}:
      a b = sum(l) sum(r)
      U32.add(a, b)

def main() -> IO(Unit):
  a = fill!(3n, 7)
  do IO<Unit>:
    u : Unit <- IO.print("host")
    IO.print(U32.show(sum!(changed(a))))

#|host
#|60
