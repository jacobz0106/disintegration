"""
Smoke tests for the SIR model.

Run from the repo root:
    python test_SIR.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from SIR import SIR_model


def test_gradients_no_attribute_error():
    """gradients() must not raise AttributeError (self.n was undefined)."""
    m = SIR_model(T=30, T0=10)
    g = m.gradients([0.28, 0.12])
    assert len(g) == 2, "gradients must return [dQ/dbeta, dQ/dgamma]"
    print(f"  PASS  gradients() returns {[round(v,6) for v in g]}")


def test_gradients_finite_difference():
    """Analytic gradients must match centred finite differences within 1e-3."""
    m = SIR_model(T=30, T0=10)
    eps = 1e-5
    for beta, gamma in [(0.28, 0.12), (0.10, 0.05), (0.32, 0.20)]:
        g = m.gradients([beta, gamma])
        fd_beta  = (m.quantity_interest([beta+eps, gamma]) -
                    m.quantity_interest([beta-eps, gamma])) / (2*eps)
        fd_gamma = (m.quantity_interest([beta, gamma+eps]) -
                    m.quantity_interest([beta, gamma-eps])) / (2*eps)
        err_beta  = abs(g[0] - fd_beta)
        err_gamma = abs(g[1] - fd_gamma)
        assert err_beta  < 1e-3, f"dQ/dbeta  error {err_beta:.2e} at ({beta},{gamma})"
        assert err_gamma < 1e-3, f"dQ/dgamma error {err_gamma:.2e} at ({beta},{gamma})"
        print(f"  PASS  FD check at beta={beta}, gamma={gamma}: "
              f"err_beta={err_beta:.2e}, err_gamma={err_gamma:.2e}")


def test_pof_initialization():
    """POFdarts.Initialize must succeed with SIR gradients (exact failure path)."""
    from POFdarts import POFdarts
    m = SIR_model(T=30, T0=10)
    critical_values = np.linspace(0.000362, 0.002885, 6)[1:-1].tolist()
    pof = POFdarts(m.quantity_interest, m.gradients, CONST_a=2,
                   critical_values=critical_values, max_iterations=50, max_miss=20)
    pof.Initialize(5, 2, [[0, 0.35], [0, 0.6]])
    assert len(pof.df) > 0, "POFdarts must place at least one sample"
    print(f"  PASS  POFdarts initialized with {len(pof.df)} points")


def test_event_estimation_sir_setup():
    """event_estimation main() SIR branch must build kde_cdf and critical_values."""
    import unittest.mock as mock
    sys.argv = ['event_estimation.py', 'SIR', 'NN', '10', 'Random']
    os.chdir(os.path.join(os.path.dirname(__file__), 'batchTask'))
    import event_estimation as ee
    with mock.patch.object(ee, 'accuracyComparison_parallel_repeat', return_value=[]) as m:
        ee.main()
        kw = m.call_args.kwargs
    assert kw['domains'] == [[0, 0.35], [0, 0.6]]
    assert kw['event'] == [[0.25, 0.35], [0.06, 0.14]]
    assert len(kw['critical_values']) == 9
    mid = float(kw['critical_values'][4])
    cdf_val = float(kw['kde_cdf'](mid))
    assert 0.0 < cdf_val < 1.0, f"kde_cdf at midpoint must be in (0,1), got {cdf_val}"
    print(f"  PASS  event_estimation SIR setup: "
          f"{len(kw['critical_values'])} critical values, kde_cdf(mid)={cdf_val:.4f}")
    os.chdir(os.path.dirname(__file__))


if __name__ == '__main__':
    tests = [
        test_gradients_no_attribute_error,
        test_gradients_finite_difference,
        test_pof_initialization,
        test_event_estimation_sir_setup,
    ]
    failed = 0
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
        except Exception as e:
            print(f"  FAIL  {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"{len(tests)-failed}/{len(tests)} passed")
    sys.exit(failed)
