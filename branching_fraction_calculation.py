import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.integrate import simps, cumtrapz
from scipy.stats import poisson


def calculate_and_plot_branching_fraction(
        B: float,
        S: float,
        model_expected_FPR: float,
        model_expected_TPR: float,
        sigma: float,
        L: float,
        n_b_to_Lb: float,
        pre_model_efficiency: float
):
    # brs = np.logspace(-6, -5, base=10, num=3000)
    brs = np.linspace(0, 1e-5, 3000)

    expected_bg = B * model_expected_FPR
    print(
        f"Expected background in signal region: {B:.3f} (before ML classifier), {expected_bg:.3f} (after ML classifier)")

    probabilities_of_seen_values = []
    probabilities_of_less_than_seen_values = []

    total_efficiency = pre_model_efficiency * model_expected_TPR
    multiplier = sigma * L * n_b_to_Lb * total_efficiency
    for br in brs:
        expected_signal_seen = multiplier * br
        mu = expected_signal_seen + expected_bg
        probability_of_seen_values = poisson.pmf(S, mu)
        probabilities_of_seen_values.append(probability_of_seen_values)
        probability_of_less_than_seen_values = poisson.cdf(S, mu)
        probabilities_of_less_than_seen_values.append(probability_of_less_than_seen_values)

    probabilities_of_seen_values = np.array(probabilities_of_seen_values)
    probabilities_of_less_than_seen_values = np.array(probabilities_of_less_than_seen_values)

    def calculate_cdf(probabilities: np.ndarray, brs):
        normalised_probabilities = probabilities / probabilities.max()
        area_under_curve = simps(y=normalised_probabilities, x=brs)
        cumulative_areas = cumtrapz(y=normalised_probabilities, x=brs, initial=0)
        cdf = cumulative_areas / area_under_curve
        return cdf

    # cdf = calculate_cdf(probabilities_of_seen_values, brs)
    for (probabilities, ylabel) in [
        (probabilities_of_seen_values, f"P({S} Events|Br=X)"),
        # (probabilities_of_less_than_seen_values, r"$\mathrm{P}(\leq" + str(S) + r" \  \mathrm{Events} | \mathrm{Br}=X )$"),
    ]:
        cdf = calculate_cdf(probabilities, brs)
        max_likelihood = brs[probabilities.argmax()]
        median_likelihood_plus_one_sigma = brs[(cdf > 0.841345).argmax()]
        median_likelihood_minus_one_sigma = brs[(cdf > 0.158655).argmax()]
        print(
            f"Max Likelihood: {max_likelihood:.3e}  (+{median_likelihood_plus_one_sigma - max_likelihood:.3e}, -{max_likelihood - median_likelihood_minus_one_sigma:.3e})")

        cl_90 = brs[(cdf > 0.9).argmax()]
        print(f"CL 90: {cl_90:.3e}")

        plt.figure(figsize=(6, 3))
        ax = plt.gca()
        ax.axvline(max_likelihood, color='C2', label=f"Max Likelihood: {max_likelihood:.3e}")

        ax.plot(brs, probabilities)

        brs_mask = brs < cl_90
        ax.fill_between(brs, probabilities, where=brs_mask, color='C1', alpha=0.3, facecolor='C1')

        ax.axvline(cl_90, color='k', label=f"90% CL: {cl_90:.3e}")

        base_power = -6

        @ticker.FuncFormatter
        def formatter(x, pos):
            latex_str = f'{x * 10 ** -base_power:.1f}' + r'\times 10^{' + str(base_power) + '}'
            return f'${latex_str}$'

        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel("Branching Franction")
        ax.set_ylabel(ylabel)
        ax.set_xlim(brs.min(), brs.max())
        ax.grid()
        ax.legend()
        plt.tight_layout()
        # plt.savefig("br_limit_lessthan.png", dpi=300)
        # plt.savefig("br_limit_equal.png", dpi=300)
        plt.savefig("br_limits.png", dpi=300)
        plt.show()


if __name__ == '__main__':
    calculate_and_plot_branching_fraction(
        B=114.35,
        S=52,
        model_expected_FPR=0.012,
        model_expected_TPR=0.562,
        sigma=280e-6 * 13.5 / 8,
        L=4e15,
        n_b_to_Lb=0.19,
        pre_model_efficiency=0.0002126 * 0.23
        # Initial cuts * approximate PID cut efficiency (calculated based on effect on J/psi)
    )

    calculate_and_plot_branching_fraction(
        B=56.37,
        S=10,
        model_expected_FPR=0.005,
        model_expected_TPR=0.558,
        sigma=280e-6 * 13.5 / 8,
        L=4e15,
        n_b_to_Lb=0.19,
        pre_model_efficiency=7.80055e-5 * 0.23
        # Initial cuts * approximate PID cut efficiency (calculated based on effect on J/psi)
    )
