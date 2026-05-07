import numpy as np
from scipy.optimize import minimize, linprog
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import sklearn
import matplotlib.pyplot as plt


class SVM_Penalized(object):
    def __init__(self, C, K, tol=0.0001, reduced=False):
        '''
        reduced: ignore the W^TW part
        '''
        self.reduced = reduced
        self.C = C
        self.K = K
        self.tol = tol
        self.alpha = []
        self.coef_ = []
        self.intercept_ = []

    def fit_SVM(self, A_train, C_train):
        svm_model = sklearn.svm.SVC(kernel='linear', C=self.C)
        svm_model.fit(A_train, C_train)
        self.coef_std = svm_model.coef_[0]
        self.intercept_std = svm_model.intercept_[0]

    def fit_nonOptimal(self, A_train, C_train):
        n = len(C_train)
        # LP: maximize sum(alpha) - (A_train @ coef_std) * C_train @ alpha
        # linprog minimizes, so negate: minimize (A_train @ coef_std)*C_train - 1 @ alpha
        c = (A_train @ self.coef_std) * C_train - 1.0
        result = linprog(
            c,
            A_eq=C_train.reshape(1, -1), b_eq=[0.0],
            bounds=[(0, self.C)] * n,
            method='highs',
        )
        if result.success:
            self.alpha = {j: result.x[j] for j in range(n)}
        else:
            print("No optimal solution found in fit_nonOptimal, with C =", self.C)
            self.alpha = {j: 0.0 for j in range(n)}

    def fit(self, A_train_original, C_train, dQ=None):
        if self.reduced:
            self.fit_reduced(A_train_original, C_train, dQ)
            return
        if dQ is None:
            raise ValueError("must provide gradients")

        scaler = StandardScaler()
        A_train = scaler.fit_transform(A_train_original)
        self.fit_SVM(A_train, C_train)

        if self.K == 0:
            self.coef_.append(self.coef_std / scaler.scale_)
            self.intercept_.append(self.intercept_std - np.dot(self.coef_std, scaler.mean_ / scaler.scale_))
            return

        n = A_train.shape[0]
        C_train = np.asarray(C_train, dtype=float)

        w_1 = self.coef_std
        dQ = [g / np.linalg.norm(g) for g in dQ]
        mean_grad = np.mean(dQ, axis=0)
        w_2 = mean_grad / np.linalg.norm(mean_grad)
        self.mean_gradient = w_2

        w_1_proj_2 = (w_1 @ w_2) / (w_2 @ w_2) * w_2
        w_1_proj_2 = w_1_proj_2 / np.linalg.norm(w_1_proj_2)

        penalty = self.K * (1.0 - (w_1 @ w_2) ** 2 / (w_1 @ w_1))

        d = w_1 - w_1_proj_2
        c_d = float(d @ d)

        if c_d < 1e-12:
            # w_1 already aligned with w_1_proj_2; t is undefined, use t=0
            self.alpha = {j: 0.0 for j in range(n)}
            t_val = 0.0
        else:
            # Precompute projections onto d and w_1_proj_2 (shape n)
            a_vec = (A_train @ d) * C_train        # dt/dalpha[k] = a_vec[k] / c_d
            e_vec = (A_train @ w_1_proj_2) * C_train
            w2_sq = float(w_1_proj_2 @ w_1_proj_2)
            wd = float(w_1_proj_2 @ d)
            b_const = wd + penalty                 # = d@w_1_proj_2 + penalty
            inv_cd = 1.0 / c_d

            # t = (a_vec @ alpha - b_const) / c_d  (linear in alpha)
            # Objective to maximize:
            #   f = sum(alpha) + 0.5*||w_t||^2 + penalty*t - alpha@e_vec - t*(a_vec@alpha)
            # Analytical gradient of -f:  e_vec + t*a_vec - 1
            def neg_objective(alpha):
                t = (a_vec @ alpha - b_const) * inv_cd
                wt_sq = w2_sq + 2.0 * t * wd + t * t * c_d
                return -(np.sum(alpha) + 0.5 * wt_sq + penalty * t
                         - alpha @ e_vec - t * (a_vec @ alpha))

            def neg_jac(alpha):
                t = (a_vec @ alpha - b_const) * inv_cd
                return e_vec + t * a_vec - 1.0

            # Feasible starting point: satisfies C_train @ alpha = 0 and box constraints.
            # x0 = C/2 * ones violates the equality for imbalanced classes.
            n_pos = int((C_train > 0).sum())
            n_neg = int((C_train < 0).sum())
            x0 = np.zeros(n)
            if n_pos > 0 and n_neg > 0:
                v = self.C / 2.0
                x0[C_train > 0] = v * min(1.0, n_neg / n_pos)
                x0[C_train < 0] = v * min(1.0, n_pos / n_neg)

            # Only enforce equality + box; the t in [0,1] inequalities can make
            # the feasible region empty when K is large, causing status-8 failures.
            # After solving, clip t to [0,1] — equivalent to projecting w_t onto
            # the segment [w_1_proj_2, w_1].
            constraints = [
                {'type': 'eq',
                 'fun': lambda a: C_train @ a,
                 'jac': lambda _: C_train},
            ]
            bounds = [(0.0, self.C)] * n

            result = minimize(
                neg_objective, x0,
                method='SLSQP', jac=neg_jac,
                bounds=bounds, constraints=constraints,
                options={'ftol': self.tol, 'maxiter': 2000},
            )

            if result.success or result.status in (8, 9):
                self.alpha = {j: result.x[j] for j in range(n)}
                t_val = float(np.clip((a_vec @ result.x - b_const) * inv_cd, 0.0, 1.0))
            else:
                print("SLSQP did not converge (status %d: %s); falling back." % (result.status, result.message))
                self.fit_nonOptimal(A_train, C_train)
                t_val = 0.0

        w_penalized_std = t_val * w_1 + (1.0 - t_val) * w_1_proj_2

        tol = 0.001
        support_vectors = [i for i in range(n) if tol < self.alpha[i] < self.C]

        if support_vectors:
            b_values = [C_train[i] - A_train[i] @ w_penalized_std for i in support_vectors]
            intercept_penalized_std = np.mean(b_values)
        else:
            intercept_penalized_std = 0.0

        self.coef_.append(w_penalized_std / scaler.scale_)
        self.intercept_.append(intercept_penalized_std - w_penalized_std @ (scaler.mean_ / scaler.scale_))

    def decision_function(self, A_train):
        return [np.sum(self.coef_[0] * A_train[j]) + self.intercept_[0] for j in range(len(A_train))]

    def predict(self, A_train):
        score = self.decision_function(A_train)
        return np.array([1 if s >= 0 else -1 for s in score])
