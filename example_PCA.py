import d0mbpy.d0mbpy as dp


a = [[0.9721474825854081, -0.9864484284317094, 0.01430094584630117],
    [-0.9864484284317094, 2.0830150213304868, -1.096566592898778],
    [0.01430094584630117, -1.096566592898778, 1.0822656470524772]]
x = dp.LinAlg(a)


# normilize data
x_norm  = (x - x.mean(axis=0)) / x.std(axis=0)

# get cov of data
x_cov = x_norm.cov()
# get eigen values and eigen vectors
e_vals, e_vec = x_cov.eigen_values()

n = 2
pca = (dp.LinAlg(e_vec.transpose()[:n])* x_norm.transpose()).transpose()

print(pca)
