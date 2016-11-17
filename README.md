### Graph Lab attempt

In the following work as the first baseline solution, after repeating your attached 
code which give me rmse score $.93$ i  then apply user-item matrix factorization algorithm from GraphLab framework, at this point i achieve rmse score $.91$ and thats all. I try many approaches, even features extraction followed with gradient boosting, blending, out of $k$ folds stacking (with $k \in {3, 5, 9}$)  and this give me nothing less than $.91$ rmse.

### My Media Lite attempt

This suitable comand-line framework have a lot of models under the hood, I try various of them, the best rmse gives me Sigmoid Item Asymmetric Factor Model, and Singular Value Decomposition with rmse $.89417$.  Last two models also blended with Sigmoid User Asymmetric Factor Model, Biased Matrix Factorization give result $.89$0 on rmse, and turn to be the best solution for me.

