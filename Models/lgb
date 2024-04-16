if LGB:
    import numpy as np
    import lightgbm as lgb
    
    lgb_params = {
        "objective": "mae",
        "n_estimators": 6000,
        "num_leaves": 256,
        "subsample": 0.6,
        "colsample_bytree": 0.8,
        "learning_rate": 0.00871,
        'max_depth': 11,
        "n_jobs": 4,
        "device": "gpu",
        "verbosity": -1,
        "importance_type": "gain",
        "reg_alpha": 0.1,
        "reg_lambda": 3.25
    }

    feature_columns = list(df_train_feats.columns)

    num_folds = 5
    fold_size = 480 // num_folds
    gap = 5 #这通常用于时间序列交叉验证中创建一个时间间隙（即间隔），防止模型在验证过程中信息泄露。

    models = []
    models_cbt = []
    scores = []

    model_save_path = 'modelitos_para_despues' 
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    date_ids = df_train['date_id'].values

    for i in range(num_folds):
        start = i * fold_size
        end = start + fold_size
        if i < num_folds - 1:  # No need to purge after the last fold
            purged_start = end - 2
            purged_end = end + gap + 2
            train_indices = (date_ids >= start) & (date_ids < purged_start) | (date_ids > purged_end)
        else:
            train_indices = (date_ids >= start) & (date_ids < end)

        test_indices = (date_ids >= end) & (date_ids < end + fold_size)
        
        gc.collect()
        
        df_fold_train = df_train_feats[train_indices]
        df_fold_train_target = df_train['target'][train_indices]
        df_fold_valid = df_train_feats[test_indices]
        df_fold_valid_target = df_train['target'][test_indices]

        print(f"Fold {i+1} Model Training")

        # Train a LightGBM model for the current fold
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            df_fold_train[feature_columns],
            df_fold_train_target,
            eval_set=[(df_fold_valid[feature_columns], df_fold_valid_target)],
            callbacks=[
                lgb.callback.early_stopping(stopping_rounds=100),
                lgb.callback.log_evaluation(period=100),
            ],
        )
     

        models.append(lgb_model)
        # Save the model to a file
        model_filename = os.path.join(model_save_path, f'doblez_{i+1}.txt')
        lgb_model.booster_.save_model(model_filename)
        print(f"Model for fold {i+1} saved to {model_filename}")

        # Evaluate model performance on the validation set
        #------------LGB--------------#
        fold_predictions = lgb_model.predict(df_fold_valid[feature_columns])
        fold_score = mean_absolute_error(fold_predictions, df_fold_valid_target)
        scores.append(fold_score)
        print(f":LGB Fold {i+1} MAE: {fold_score}")

        # Free up memory by deleting fold specific variables
        del df_fold_train, df_fold_train_target, df_fold_valid, df_fold_valid_target
        gc.collect()

    # Calculate the average best iteration from all regular folds
    average_best_iteration = int(np.mean([model.best_iteration_ for model in models]))

    # Update the lgb_params with the average best iteration
    final_model_params = lgb_params.copy()

    # final_model_params['n_estimators'] = average_best_iteration
    # print(f"Training final model with average best iteration: {average_best_iteration}")

    # Train the final model on the entire dataset
    num_model = 1

    for i in range(num_model):
        final_model = lgb.LGBMRegressor(**final_model_params)
        final_model.fit(
            df_train_feats[feature_columns],
            df_train['target'],
            callbacks=[
                lgb.callback.log_evaluation(period=100),
            ],
        )
        # Append the final model to the list of models
        models.append(final_model)
