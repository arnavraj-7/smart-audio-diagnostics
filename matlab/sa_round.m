function r = sa_round(x)
% Round half to even, exactly like Python's round(). MATLAB's round() sends
% halves away from zero, so a 10 ms hop at 22 050 Hz (220.5 samples) became
% 221 here and 220 in Python, and every music feature drifted by ~0.1 %.
r = round(x);
half = abs(x - fix(x)) == 0.5;
r(half) = 2 * round(x(half) / 2);
end
