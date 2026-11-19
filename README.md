# Problem-Set-4
**Model:** Claude ~ Opus 5.5 ; **Date:** 11/18

***Prompt:*** [Prep] Act as a machine learning engineer who diagnoses model
performance problems in production.

[Request] I have a store-level revenue regression with 400 rows and
about 12 predictors. Write Python that produces a learning curve --
training and validation error as a function of training set size --
and interprets it to distinguish a high-bias problem from a
high-variance one.

[Iterate] scikit-learn, numpy and matplotlib. Reuse my X and y. Plot
both curves on one axis with a shaded band for variability across
folds.

[Mechanism Check] Add comments explaining what the GAP between the two
curves indicates versus what the LEVEL both converge to indicates, and
what each implies about whether collecting more data would help.

[Evaluate] Run it on my data and tell me whether more stores or better
features is the higher-return investment.

***Reply:***

I couldn’t run it on your data because no file was attached to this chat, so X and y don’t exist on my end. I tested the script on made-up data of the same shape (400 rows, 12 predictors) to confirm it runs. If you upload your data file, or paste the printed output, I’ll give you the diagnosis.

What the script does: It fits a linear regression on growing slices of your data (10% up to 100%) with 5-fold cross-validation. It plots the average training error and validation error, with shaded bands showing the spread across folds. Because each fold holds out a fifth of the data, the largest training size is 320 stores, not 400. The gray dashed line shows the error you’d get by just predicting average revenue for every store, which gives you a reference for whether an error is “high”.

How to decide between more stores and better features, using the printed numbers:

More stores is the better investment if the gap is large (say, a third or more of validation error) and the validation curve is still clearly sloping down at the right edge. The model is overfitting, and more data is still paying off.
Better features is the better investment if the gap is small, both curves have flattened, and they sit at a high level, meaning the model removes only a modest share of the guess-the-average error. Both curves have stopped moving, so another 400 stores would land you in the same place. You need new information in the predictors.
If both the gap and the level are small, the model is already doing well, and neither investment is urgent.

These cutoffs are rules of thumb, not formal tests. The wide shaded bands you’ll likely see at small sizes are normal with only 400 rows.

For example, the test run on fake data showed a gap of only 4% and almost no drop at the last step. The model still removed just 56% of the guess-the-average error. That pattern points to bias, so better features would be the call.

**Code:** import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve, KFold

# Uses your existing X (400 x ~12) and y (store revenue)

# Fit the model on 10%, 20%, ..., 100% of the training folds.
# Shuffle in case the rows are sorted (e.g. by region or store size).
cv = KFold(n_splits=5, shuffle=True, random_state=0)
sizes, train_scores, val_scores = learning_curve(
    LinearRegression(), X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=cv, scoring="neg_root_mean_squared_error")

# sklearn reports NEGATIVE RMSE, so flip the sign to get error
train_err = -train_scores
val_err = -val_scores

# Mean and spread across the 5 folds at each training size
train_mean, train_sd = train_err.mean(axis=1), train_err.std(axis=1)
val_mean, val_sd = val_err.mean(axis=1), val_err.std(axis=1)

plt.plot(sizes, train_mean, "o-", label="Training error")
plt.fill_between(sizes, train_mean - train_sd, train_mean + train_sd, alpha=0.2)
plt.plot(sizes, val_mean, "o-", label="Validation error")
plt.fill_between(sizes, val_mean - val_sd, val_mean + val_sd, alpha=0.2)
plt.axhline(np.std(y), linestyle="--", color="gray", label="Guess the average")
plt.xlabel("Number of stores used for training")
plt.ylabel("RMSE (revenue units)")
plt.title("Learning curve")
plt.legend()
plt.show()

# ---------------- How to read it ----------------
# GAP (validation error minus training error) = VARIANCE.
#   A big gap means the model fits the stores it trained on much better
#   than new stores, i.e. it is learning noise. If the validation curve is
#   still falling at the right edge, the gap is closing as data grows, so
#   MORE STORES would help.
#
# LEVEL (where both curves end up) = BIAS.
#   If both curves flatten out close together but at a HIGH error (close to
#   the "guess the average" line), the model is too simple for the problem.
#   More rows will not help, because both curves have already stopped
#   moving. BETTER FEATURES (new variables, interactions, transformations)
#   are what lower the level.

gap = val_mean[-1] - train_mean[-1]
still_falling = val_mean[-2] - val_mean[-1]   # drop over the last step

print(f"Final training RMSE:    {train_mean[-1]:.2f}")
print(f"Final validation RMSE:  {val_mean[-1]:.2f}")
print(f"Gap:                    {gap:.2f}  ({gap / val_mean[-1]:.0%} of validation error)")
print(f"Validation drop, last step: {still_falling:.2f}")
print(f"Guess-the-average RMSE: {np.std(y):.2f}")
print(f"Share of error removed by model: {1 - val_mean[-1] / np.std(y):.0%}")

