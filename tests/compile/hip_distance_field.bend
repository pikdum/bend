import Base

type Grid is Data:
  Cell{v: U32}
  Quad{a: Grid, b: Grid, c: Grid, d: Grid}

def width() -> U32:
  64

def depth() -> Nat:
  6n

def sweeps() -> Nat:
  128n

def choose(b: Bool, x: U32, y: U32) -> U32:
  match b:
    case True{}:
      x
    case False{}:
      y

def blocked(+x: U32, +y: U32) -> Bool:
  Bool.and(U32.is_eq(x, U32.div(width(), 2)), U32.is_ne(y, U32.div(width(), 2)))

def seed(+d: Nat, +x: U32, +y: U32, +blank: Bool) -> Grid:
  match d:
    case 0n:
      Cell{choose(Bool.or(blank, Bool.or(U32.is_ne(x, 0), U32.is_ne(y, 0))), 1000000, 0)}
    case 1n+p:
      +h = U32.shln(1, p)
      a b c d = seed(p, x, y, blank) seed(p, U32.add(x, h), y, blank) seed(p, x, U32.add(y, h), blank) seed(p, U32.add(x, h), U32.add(y, h), blank)
      Quad{a, b, c, d}

def step(+level: Nat, +x: U32, +y: U32, f: Grid, n: Grid, s: Grid, w: Grid, e: Grid) -> Grid:
  match level:
    case 0n:
      match f n s w e:
        case Cell{v} Cell{vn} Cell{vs} Cell{vw} Cell{ve}:
          Cell{choose(blocked(x, y), 1000000, U32.min(v, U32.inc(U32.min(U32.min(vn, vs), U32.min(vw, ve)))))}
        case _ _ _ _ _:
          Cell{1000000}
    case 1n+p:
      match f n s w e:
        case Quad{+a, +b, +c, +d} Quad{na, nb, nc, nd} Quad{sa, sb, sc, sd} Quad{wa, wb, wc, wd} Quad{ea, eb, ec, ed}:
          +h = U32.shln(1, p)
          aa bb cc dd = step(p, x, y, a, nc, c, wb, b) step(p, U32.add(x, h), y, b, nd, d, a, ea) step(p, x, U32.add(y, h), c, a, sa, wd, d) step(p, U32.add(x, h), U32.add(y, h), d, b, sb, c, ec)
          Quad{aa, bb, cc, dd}
        case _ _ _ _ _:
          Cell{1000000}

def solve(n: Nat, f: Grid, +blank: Grid) -> Grid:
  match n:
    case 0n:
      f
    case 1n+p:
      solve(p, step(depth(), 0, 0, f, blank, blank, blank, blank), blank)

def expected(+x: U32, +y: U32) -> U32:
  +m = U32.div(width(), 2)
  delta = U32.sub(U32.max(y, m), U32.min(y, m))
  choose(blocked(x, y), 1000000, choose(U32.is_lt(x, m), U32.add(x, y), U32.add(U32.add(x, m), delta)))

type Stats is Data:
  Stats{sum: U32, errors: U32}

def merge(a: Stats, b: Stats, c: Stats, d: Stats) -> Stats:
  Stats{av, ae} = a
  Stats{bv, be} = b
  Stats{cv, ce} = c
  Stats{dv, de} = d
  Stats{U32.add(U32.add(av, bv), U32.add(cv, dv)), U32.add(U32.add(ae, be), U32.add(ce, de))}

def validate(f: Grid, +x: U32, +y: U32, +size: U32) -> Stats:
  match f:
    case Cell{+v}:
      Stats{U32.mul(U32.inc(U32.add(U32.mul(y, width()), x)), v), choose(U32.is_eq(v, expected(x, y)), 0, 1)}
    case Quad{a, b, c, d}:
      +h = U32.shr(size)
      aa bb cc dd = validate(a, x, y, h) validate(b, U32.add(x, h), y, h) validate(c, x, U32.add(y, h), h) validate(d, U32.add(x, h), U32.add(y, h), h)
      merge(aa, bb, cc, dd)

def report(s: Stats) -> String:
  Stats{sum, errors} = s
  U32.show(sum) ++ " " ++ U32.show(errors)

def compute() -> Stats:
  validate(solve(sweeps(), seed(depth(), 0, 0, False{}), seed(depth(), 0, 0, True{})), 0, 0, width())

def main() -> IO(Unit):
  IO.print(report(compute!()))

#|839108608 0
