from config import Config
from anilist_requests import *
from machine_learing import *
#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score


user_name = Config.USER
split_ration = Config.SPLIT_RATIO

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # 1) Get List of Anime and Manga from AnilistAPI
    anilist_ratted, message = get_anilist_rated(user_name) # user = me , format = default
    #print(f"1) Data Returned from Anilist: {anilist_ratted}")
    print(f" # MESSAGE : {message}")

    # 2) Prepare Data format for learing algorithm
    formated_anilist = format_data(anilist_ratted)
    print(f"2) Formated Data:")

    # 3) Spit Data
    X_train, X_test, y_train, y_test = split_data(data=formated_anilist, split_ratio=split_ration)

    # 4) Train Model
    # Model 1: RandomForestRegressor
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    print(rf_predictions)
    # Model 2: GradientBoostingRegressor
    gb_model = GradientBoostingRegressor()
    gb_model.fit(X_train, y_train)
    gb_predictions = gb_model.predict(X_test)

    # # Model 3: XGBRegressor (XGBoost)
    # xgb_model = XGBRegressor()
    # xgb_model.fit(X_train, y_train)
    # xgb_predictions = xgb_model.predict(X_test)

    # 5) Evaluate models
    models = [rf_model, gb_model]
    model_names = ['RandomForestRegressor', 'GradientBoostingRegressor', 'XGBRegressor']

    for i, model in enumerate(models):
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        explained_variance = explained_variance_score(y_test, predictions)

        print(f"Metrics for {model_names[i]}:")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"R-squared (R2): {r2}")
        print(f"Explained Variance Score: {explained_variance}")
        print("\n")

    # 6) Use
    anilist_planned, message = get_anilist_planning(user_name)
    print(message)
    anilist_planned_formated = format_data_plan(anilist_planned)
    anilist_planned_original = format_data_prediction(anilist_planned)
    model_results(rf_model, anilist_planned_formated, anilist_planned_original, X_train)
    # Use the model on the

    # :)


