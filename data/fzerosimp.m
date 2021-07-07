function b = fzerosimp(xl,xu)
a = xl; b = xu; fa = f(a); fb = f(b);
c = a; fc = fa; d = b - c; e = d;
while (1)
  if fb == 0, break, end
  if sign(fa) == sign(fb) %If needed, rearrange points
    a = c; fa = fc; d = b - c; e = d;
  end
  if abs(fa) < abs(fb)
    c = b; b = a; a = c;
    fc = fb; fb = fa; fa = fc;
  end
  m = 0.5*(a - b); %Termination test and possible exit
  tol = 2 * eps * max(abs(b), 1);
  if abs(m) <= tol | fb == 0.
    break
  end
  %Choose open methods or bisection
  if abs(e) >= tol & abs(fc) > abs(fb)
    s = fb/fc;
    if a == c %Secant method
      p = 2*m*s;
      q = 1 - s;
    else %Inverse quadratic interpolation
      q = fc/fa; r = fb/fa;
      p = s * (2*m*q * (q - r) - (b - c)*(r - 1));
      q = (q - 1)*(r - 1)*(s - 1);
    end
    if p > 0, q = -q; else p = -p; end;
    if 2*p < 3*m*q - abs(tol*q) & p < abs(0.5*e*q)
      e = d; d = p/q;
    else
      d = m; e = m;
    end
  else %Bisection
    d = m; e = m;
  end
  c = b; fc = fb;
  if abs(d) > tol, b=b+d; else b=b-sign(b-a)*tol; end
  fb = f(b);
end