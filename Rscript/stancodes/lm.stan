data {
  int<lower=1> N; // Number of data
  int<lower=1> D; // Number of covariates
  matrix[N, D] X;
  real y[N];

  int<lower=1> N_test;
  matrix[N_test, D] X_test;
  real y_test[N_test];
}

parameters {
  real alpha;           
  vector[D] beta;
  real<lower=0> sigma;
}

transformed parameters {     
  vector[N] mu;

  mu = alpha + X * beta;
}

model {
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);

  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N_test] y_pred;
  vector[N_test] indiv_squared_errors;

  real <lower = 0> sum_of_squares;
  real <lower = 0> root_mean_squared_error;

  vector[N_test] mu_new;

  {
    mu_new = alpha + X_test * beta;
    
    for (n in 1:N_test)  {
      y_pred[n] = normal_rng(mu_new[n], sigma);
      indiv_squared_errors[n] = (y_test[n] - y_pred[n])^2;
    }
  }
  // sum of squares, data set size and scale specific
  sum_of_squares = sum(indiv_squared_errors);

  // divide by number of new / test points and sqrt for RMSE
  root_mean_squared_error = sqrt(sum_of_squares / N_test);
  
}
