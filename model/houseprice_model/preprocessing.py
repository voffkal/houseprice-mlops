"""Препроцессинг: по-колоночный ColumnTransformer и top_pipe (порт из Houseprice.ipynb).

Слоты CT намеренно содержат значения по умолчанию из тетрадки; победившие
варианты серча применяются поверх через set_params в builders/.

Отклонение от тетрадки (2026-07-15): unseen='ignore'→'encode' во всех энкодерах —
'ignore' даёт NaN на незнакомых категориях (падение в CV и в проде);
'encode' реализует исходный замысел «не падать» (-1/0, как в MapEncoder).
"""

import feature_engine.encoding as fe
import feature_engine.imputation as mdi
import feature_engine.outliers as outl
import sklearn.preprocessing as pp
from feature_engine.selection import DropConstantFeatures, DropFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from houseprice_model.mappings import NA_FILLERS, QUAL_MAPS
from houseprice_model.transformers import (
    ConditionFlags,
    GroupMedianImputer,
    MapEncoder,
    SumFeatures,
)


def build_ct() -> ColumnTransformer:
    ct = ColumnTransformer([

        ('Electrical', Pipeline([
            ('impute', mdi.CategoricalImputer(imputation_method='frequent', ignore_format=True)),
            ('rare',   'passthrough'),
            ('encode', fe.OrdinalEncoder(encoding_method='arbitrary', unseen='encode', ignore_format=True)),
            ('drop', 'passthrough')
        ]), ['Electrical']),

        ('MSSubClass', Pipeline([
            ('impute', mdi.CategoricalImputer(imputation_method='frequent', ignore_format=True)),
            ('rare',   'passthrough'),
            ('encode', fe.CountFrequencyEncoder(encoding_method='frequency', unseen='encode', ignore_format=True)),
            ('drop', 'passthrough')
        ]), ['MSSubClass']),

        ('MSZoning', Pipeline([
            ('impute', mdi.CategoricalImputer(imputation_method='frequent', ignore_format=True)),
            ('rare',   'passthrough'),
            ('encode', 'passthrough'),
            ('drop', 'passthrough')
        ]), ['MSZoning']),

        ('LotFrontage', Pipeline([
            ('outlier', 'passthrough'),
            ('transform', 'passthrough'),
            ('discretiser', 'passthrough'),
            ('drop', 'passthrough')
        ]), ['LotFrontage']),

        ('LotArea', Pipeline([
            ('outlier', 'passthrough'),
            ('transform', 'passthrough'),
            ('discretiser', 'passthrough'),
            ('drop', 'passthrough')
        ]), ['LotArea']),

        ('LotShape', Pipeline([
            ('rare',   'passthrough'),
            ('encode', fe.OrdinalEncoder(encoding_method='arbitrary', unseen='encode', ignore_format=True)),
            ('drop', 'passthrough')
        ]), ['LotShape']),

        ('LandContour', Pipeline([
            ('rare',   'passthrough'),
            ('encode', fe.OrdinalEncoder(encoding_method='arbitrary', unseen='encode', ignore_format=True)),
            ('drop', 'passthrough')
        ]), ['LandContour']),

        ('LotConfig', Pipeline([
            ('rare',   'passthrough'),
            ('encode', fe.OrdinalEncoder(encoding_method='arbitrary', unseen='encode', ignore_format=True)),
            ('drop', 'passthrough')
        ]), ['LotConfig']),

        ('LandSlope', Pipeline([
            ('encode', fe.OrdinalEncoder(encoding_method='arbitrary', unseen='encode', ignore_format=True)),
            ('drop', 'passthrough')
        ]), ['LandSlope']),

        ('Neighborhood', Pipeline([
            ('rare',   'passthrough'),
            ('encode', 'passthrough')
        ]), ['Neighborhood']),

        ('BldgType', Pipeline([
            ('rare',   'passthrough'),
            ('encode', fe.OneHotEncoder(drop_last=True, drop_last_binary=True, ignore_format=True)),
            ('drop', 'passthrough')
        ]), ['BldgType']),

        ('HouseStyle', Pipeline([
            ('rare',   fe.RareLabelEncoder(tol=0.04, n_categories=2, ignore_format=True)),
            ('encode', fe.OneHotEncoder(drop_last=True, ignore_format=True)),
            ('drop', 'passthrough')
        ]), ['HouseStyle']),

        ('OverallQual', 'passthrough', ['OverallQual']),

        ('OverallCond', Pipeline([('drop','passthrough')]), ['OverallCond']),

        ('has_positive',  'passthrough', ['has_positive']),

        ('near_railroad', 'passthrough', ['near_railroad']),

        ('near_road',     'passthrough', ['near_road']),

        ('YearBuilt', 'passthrough', ['YearBuilt']),

        ('MoSold', Pipeline([
            #('rare',   'passthrough'),
            ('encode', 'passthrough'),
            ('drop', 'passthrough')
        ]), ['MoSold']),

        ('YrSold', Pipeline([
            ('encode', 'passthrough'),
            ('drop', 'passthrough')
        ]), ['YrSold']),

        #('RoofStyle', 'passthrough', ['RoofStyle']),

        #('RoofMatl', 'passthrough', ['RoofMatl']),

        ('MasVnrType', 'passthrough', ['MasVnrType']),

        ('MasVnrArea', Pipeline([
            ('imputer', mdi.MeanMedianImputer(imputation_method='median')),
            ('outlier', outl.Winsorizer(capping_method='gaussian')),
            ('drop', 'passthrough')
        ]), ['MasVnrArea']),

        ('ExterQual', 'passthrough', ['ExterQual']),

        ('ExterCond', 'passthrough', ['ExterCond']),

        ('Exterior1st', Pipeline([
            ('imputer', mdi.CategoricalImputer(imputation_method='frequent', ignore_format=True)),
            ('rare',   fe.RareLabelEncoder(tol=0.03, n_categories=4, ignore_format=True)),
            ('encode', fe.OneHotEncoder(drop_last=True, ignore_format=True)),
        ]), ['Exterior1st']),

        #('Exterior2nd', 'passthrough', ['Exterior2nd']),

        ('Foundation', 'passthrough', ['Foundation']),

        #('Heating', 'passthrough', ['Heating']),

        ('HeatingQC', 'passthrough', ['HeatingQC']),

        ('CentralAir', Pipeline([
            ('encode', fe.OneHotEncoder(drop_last=True, drop_last_binary=True, ignore_format=True)),
            ('drop', 'passthrough')
        ]), ['CentralAir']),

        ('TotalPorchSF', SumFeatures(['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', '3SsnPorch'], 'TotalPorchSF'),
         ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', '3SsnPorch']),

        ('TotalBsmtSF', Pipeline([
            ('imputer',   mdi.ArbitraryNumberImputer(arbitrary_number=0)),  # NaN=нет подвала=0 (Kaggle test)
            ('outlier',   'passthrough'),   # серч включит Winsorizer для Ridge
            ('transform', 'passthrough'),
        ]), ['TotalBsmtSF']),

        ('BsmtFinSF1', Pipeline([
            ('imputer',   mdi.ArbitraryNumberImputer(arbitrary_number=0)),
            ('outlier',   'passthrough'),
            ('transform', 'passthrough'),
        ]), ['BsmtFinSF1']),

        ('BsmtQual',     'passthrough', ['BsmtQual']),

        ('BsmtExposure', 'passthrough', ['BsmtExposure']),

        ('BsmtFinType1', 'passthrough', ['BsmtFinType1']),

        ('BsmtCond',     'passthrough', ['BsmtCond']),

        #('BsmtFinSF2', 'passthrough', ['BsmtFinSF2']),

        #('BsmtUnfSF', 'passthrough', ['BsmtUnfSF']),

        #('BsmtFinType2', 'passthrough', ['BsmtFinType2']),

        ('GrLivArea', Pipeline([
            ('outlier',   'passthrough'),   # серч -> Winsorizer для Ridge (аутлайер 4676)
            ('transform', 'passthrough'),
        ]), ['GrLivArea']),

        ('1stFlrSF', Pipeline([
            ('outlier',   'passthrough'),
            ('transform', 'passthrough'),
        ]), ['1stFlrSF']),

        ('2ndFlrSF', Pipeline([
            ('outlier',   'passthrough'),
            ('drop', 'passthrough')
        ]), ['2ndFlrSF']),     # 56% нулей -> transform не нужен

        #('LowQualFinSF', 'passthrough', ['LowQualFinSF']),

        ('KitchenQual', 'passthrough', ['KitchenQual']),

        ('TotalBath',
        SumFeatures(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'],
                    'TotalBath',
                    weights={'HalfBath': 0.5, 'BsmtHalfBath': 0.5}),
        ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']),

        #('TotRmsAbvGrd', 'passthrough', ['TotRmsAbvGrd']),

        ('BedroomAbvGr', Pipeline([
            ('encode', 'passthrough'),
        ]), ['BedroomAbvGr']),

        #('KitchenAbvGr', 'passthrough', ['KitchenAbvGr']),

        ('FireplaceQu', 'passthrough', ['FireplaceQu']),

        #('Functional', 'passthrough', ['Functional']),

        ('GarageCars', Pipeline([
            ('imputer', mdi.ArbitraryNumberImputer(arbitrary_number=0)),
        ]), ['GarageCars']),

        ('GarageArea', Pipeline([
            ('imputer', mdi.ArbitraryNumberImputer(arbitrary_number=0)),
            ('outlier', 'passthrough'),
        ]), ['GarageArea']),

        ('GarageType',   'passthrough', ['GarageType']),

        ('GarageFinish', 'passthrough', ['GarageFinish']),

        ('GarageQual',   'passthrough', ['GarageQual']),

        #('GarageYrBlt', 'passthrough', ['GarageYrBlt']),

        #('GarageCond',   'passthrough', ['GarageCond']),

        ('PavedDrive', 'passthrough', ['PavedDrive']),

        ('SaleCondition', Pipeline([
            ('imputer', mdi.CategoricalImputer(imputation_method='frequent', ignore_format=True)),
            ('rare',   'passthrough'),
            ('encode', 'passthrough'),
        ]), ['SaleCondition']),

        ('SaleType', Pipeline([
            ('imputer', mdi.CategoricalImputer(imputation_method='frequent', ignore_format=True)),
            ('rare',   'passthrough'),
            ('encode', 'passthrough'),
        ]), ['SaleType']),

        ('HouseAge', 'passthrough',
        ['YrSold', 'YearBuilt']),

        ('poly', 'passthrough', ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'YearBuilt']),

    ], remainder='drop', verbose_feature_names_out=False)
    return ct


def build_top_pipe() -> Pipeline:
    top_pipe = Pipeline([
        ('drop_directly', DropFeatures(features_to_drop=[
            # No informative variables
            'Id',
            #'MoSold',
            #'YrSold',

            # NaN > 80%
            'MiscFeature',
            'Alley',
            'Fence',

            # Const
            'Street',
            'Utilities',
            'PoolArea',
            'PoolQC',
            'MiscVal',

            ])),
        ('map', MapEncoder(QUAL_MAPS, NA_FILLERS, unseen=-1)),
        ('lotfrontage_imputer', GroupMedianImputer(target='LotFrontage', group='Neighborhood')),
        # NaN = «объекта нет» -> 0 (та же логика, что NA_FILLERS='NoBasement'/'NoGarage').
        # Импутация ДО ct обязательна: CT раздаёт слотам сырые колонки, и poly-слот
        # получал NaN в обход слотовых импьютеров (kaggle-экзамен 2026-07-17, Id 2121/2577).
        ('impute_no_object', mdi.ArbitraryNumberImputer(
            arbitrary_number=0,
            variables=['TotalBsmtSF', 'GarageCars', 'GarageArea', 'BsmtFinSF1'])),
        # Защитный: в train/test не встречается, но контракт «модель не падает на NaN»
        ('impute_core', mdi.MeanMedianImputer(
            imputation_method='median',
            variables=['OverallQual', 'GrLivArea', 'YearBuilt'])),
        ('condition', ConditionFlags()),
        ("ct", build_ct()),
        ('drop_constant', DropConstantFeatures(tol=0.98)),
        ('scaler', pp.MinMaxScaler()),
        #('drop_low_variance', VarianceThreshold(threshold=0.01)),
        #('drop_correlated', DropCorrelatedFeatures(threshold=0.85)),
        #('selector', SelectKBest(score_func=f_regression, k=50)),
        ('model', None)
    ])
    return top_pipe
