#
# HW 4 problem 4
# Yinhao Zhu
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mv_norm
from scipy.stats import invgamma
from analysis import moving_means, auto_covariance

# load data
data = np.loadtxt("BrakingData.txt")
x_data = np.expand_dims(data[:, 0], 1)
y_data = np.expand_dims(data[:, 1], 1)
print(data.shape)
# design matrix
Phi = np.concatenate((np.ones(x_data.shape), x_data, x_data ** 2), axis=1)
# MLE
beta_mle = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T).dot(y_data)
sigma2_mle = np.sum((y_data - Phi.dot(beta_mle)) ** 2) / x_data.shape[0]
print("MLE of beta's: \n{}".format(beta_mle))
print("MLE of sigma^2: \n{}".format(sigma2_mle))


# proposal distribution
inv_Fisher_info_matrix = sigma2_mle * np.linalg.inv(Phi.T.dot(Phi))
beta_proposal = mv_norm(mean=beta_mle.squeeze(), cov=inv_Fisher_info_matrix)
a = 23.5
b = 1.0/0.000185
sigma2_proposal = invgamma(a=a, scale=b)

print("proposal distribution:")
print("For beta: q(beta) ~ Norm(beta | beta_MLE, inv_Fisher_info_matrix)")
print("For sigma^2: q(sigma^2) ~ InvGamma(sigma^2 | shape=23.5, scale=sigma^2_MLE * 50 / 2")

def normal_prior():
    print("Normal prior case")

    n_samples = 50000
    n_burn_in = 5000

    samples_beta = np.zeros((n_samples, 3))
    samples_sigma2 = np.zeros((n_samples))
    samples_beta[0] = beta_mle[:, 0]
    samples_sigma2[0] = sigma2_mle

    def posterior(beta, sigma2):
        return np.exp(-0.5 * np.sum((y_data.squeeze() - Phi.dot(beta)) ** 2) / sigma2) \
               / sigma2 ** (x_data.shape[0] / 2)

    for i in range(1, n_samples):
        # indpendence sampling
        beta = beta_proposal.rvs()
        sigma2 = sigma2_proposal.rvs()
        ratio = posterior(beta, sigma2) / posterior(samples_beta[i - 1], samples_sigma2[i - 1]) \
                * beta_proposal.pdf(samples_beta[i - 1]) * sigma2_proposal.pdf(samples_sigma2[i - 1]) \
                / beta_proposal.pdf(beta) / sigma2_proposal.pdf(sigma2)
        prob_accept = np.minimum(1, ratio)

        if np.random.uniform() < prob_accept:
            samples_beta[i] = beta
            samples_sigma2[i] = sigma2
        else:
            samples_beta[i] = samples_beta[i - 1]
            samples_sigma2[i] = samples_sigma2[i - 1]
        if i % 1000 == 0:
            print(i)

    print("sample means of beta: \n{}".format(np.mean(samples_beta, axis=0)))
    print("sample means of sigma^2: \n{}".format(np.mean(samples_sigma2)))
    # plot
    fig, axes = plt.subplots(2, 2)
    plt.suptitle("Prob 4. Normal prior")
    for i, ax in enumerate(fig.axes):
        if not i == 3:
            ax.hist(samples_beta[n_burn_in:, i], 50, normed=1)
            ax.set_title(r"Histogram of MC samples for $\beta_{}$".format(i))

    axes[1, 1].hist(samples_sigma2[n_burn_in:], 50, normed=1)
    axes[1, 1].set_title(r"Histogram of MC samples for $\sigma^2$")
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.subplots_adjust(top=0.85)
    # plt.show()
    plt.savefig("figs/p4_normal_prior.png", bbox_inches='tight')

    # analysis
    means_beta = moving_means(samples_beta[n_burn_in:])
    means_sigma2 = moving_means(samples_sigma2[n_burn_in:])
    auto_cov_beta = auto_covariance(samples_beta[n_burn_in:])
    auto_cov_sigma2 = auto_covariance(samples_sigma2[n_burn_in:])
    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(122)
    for i in range(3):
        ax1.semilogx(range(len(means_beta)), means_beta[:, i],
                     label=r"$\beta_{}$".format(i))
    ax2.semilogx(range(len(means_sigma2)), means_sigma2, label=r"$\sigma^2$")
    ax1.set_title("Moving means of samples")
    ax2.set_title("Moving means of samples")
    ax1.legend()
    ax2.legend()

    for i in range(3):
        ax3.plot(range(len(auto_cov_beta)), auto_cov_beta[:, i],
                     label=r"$\beta_{}$".format(i))
    ax3.plot(range(len(auto_cov_sigma2)), auto_cov_sigma2, label=r"$\sigma^2$")
    ax3.set_title("Auto-covariance of samples")
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    ax3.legend()
    plt.savefig("figs/p4_normal_prior_analysis.png", bbox_inches='tight')
    print("Normal prior: done plotting analysis figure")


# beta_mle = beta_mle
# sigma2_mle = sigma2_mle
# # proposal distribution
# inv_Fisher_info_matrix = sigma2_mle * np.linalg.inv(Phi.T.dot(Phi))
# beta_proposal = mv_norm(mean=beta_mle.squeeze(), cov=inv_Fisher_info_matrix)
# a = 4
# b = sigma2_mle * (a - 1)
# sigma2_proposal = invgamma(a=a, scale=b)


def student_t_prior():
    print("Student-t prior case")
    # student-t likelihood
    n_samples = 50000
    n_burn_in = 5000

    samples_beta = np.zeros((n_samples, 3))
    samples_sigma2 = np.zeros((n_samples))
    samples_beta[0] = beta_mle[:, 0]
    samples_sigma2[0] = sigma2_mle

    nu = 4
    y = y_data.squeeze()
    x = x_data.squeeze()
    def posterior_t(beta, sigma2):
        return np.prod(((y - Phi.dot(beta)) ** 2 / sigma2 / nu + 1) ** (-(nu + 1) / 2)) \
               / sigma2 ** (x.shape[0] / 2)

    for i in range(1, n_samples):
        # indpendence sampling
        beta = beta_proposal.rvs()
        sigma2 = sigma2_proposal.rvs()
        ratio = posterior_t(beta, sigma2) / posterior_t(samples_beta[i - 1], samples_sigma2[i - 1]) \
                * beta_proposal.pdf(samples_beta[i - 1]) * sigma2_proposal.pdf(samples_sigma2[i - 1]) \
                / beta_proposal.pdf(beta) / sigma2_proposal.pdf(sigma2)
        prob_accept = np.minimum(1, ratio)

        if np.random.uniform() < prob_accept:
            samples_beta[i] = beta
            samples_sigma2[i] = sigma2
        else:
            samples_beta[i] = samples_beta[i - 1]
            samples_sigma2[i] = samples_sigma2[i - 1]
        if i % 1000 == 0:
            print(i)

    print("sample means of beta: \n{}".format(np.mean(samples_beta, axis=0)))
    print("sample means of sigma^2: \n{}".format(np.mean(samples_sigma2)))
    # plot
    fig, axes = plt.subplots(2, 2)
    plt.suptitle("Prob 4. Student-t prior")
    for i, ax in enumerate(fig.axes):
        if not i == 3:
            ax.hist(samples_beta[n_burn_in:, i], 50, normed=1)
            ax.set_title(r"Histogram of MC samples for $\beta_{}$".format(i))

    axes[1, 1].hist(samples_sigma2[n_burn_in:], 50, normed=1)
    axes[1, 1].set_title(r"Histogram of MC samples for $\sigma^2$")
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.subplots_adjust(top=0.85)
    # plt.show()
    plt.savefig("figs/p4_student_t_prior_2.png", bbox_inches='tight')

    # analysis
    means_beta = moving_means(samples_beta[n_burn_in:])
    means_sigma2 = moving_means(samples_sigma2[n_burn_in:])
    auto_cov_beta = auto_covariance(samples_beta[n_burn_in:])
    auto_cov_sigma2 = auto_covariance(samples_sigma2[n_burn_in:])

    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(122)

    # fig, axes = plt.subplots(1, 2)
    for i in range(3):
        ax1.semilogx(range(len(means_beta)), means_beta[:, i],
                     label=r"$\beta_{}$".format(i))
    ax2.semilogx(range(len(means_sigma2)), means_sigma2, label=r"$\sigma^2$")
    ax1.set_title("Moving means of samples")
    ax2.set_title("Moving means of samples")
    ax1.legend()
    ax2.legend()

    for i in range(3):
        ax3.plot(range(len(auto_cov_beta)), auto_cov_beta[:, i],
                     label=r"$\beta_{}$".format(i))
    ax3.plot(range(len(auto_cov_sigma2)), auto_cov_sigma2, label=r"$\sigma^2$")
    ax3.set_title("Auto-covariance of samples")
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    ax3.legend()
    plt.savefig("figs/p4_student_t_analysis.png", bbox_inches='tight')
    print("Student-t prior: done plotting analysis figure")

# main
normal_prior()
student_t_prior()

