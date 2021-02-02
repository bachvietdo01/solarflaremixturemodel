library(rstan)

path = '~/Documents/UM Stats PhD/research/yang_solarflare/solarflaremixturemodel'
setwd(path)

# load data
X = read.csv('data/Xflarespca3.csv')
y = read.csv('data/yflares.csv')[,1]

X_train = X[1:220,]
X_test = X[220:275,]
y_train = y[1:220]
y_test = y[220:275]


# load stan file
model_file =  'stancodes/lm_mm_qr.stan'
stanc(model_file)


# fit models
stan_data = within(list(), {
                 N = nrow(X_train)
                 D = ncol(X_train)
                 K = 5
                 X = X_train
                 y = y_train
                 X_test = X_test
                 y_test = y_test
                 N_test = nrow(X_test)
                 dir_alpha = rep(1/K, K)
                 I = diag(D)
})


stan_ols <- stan(file = model_file, data = stan_data,
                 warmup = 5000, iter = 10000, chains = 4, cores = 2, thin = 5)

traceplot(stan_ols, pars = c('theta'))

params = extract(stan_ols)
mean(params$root_mean_squared_error)
