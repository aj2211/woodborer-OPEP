#!/usr/bin/env python3
from pest_predictions_utils import *

inputFileName = 'training_data.xlsx'
outputFileName = "entry_sheet_scolytinae.xlsx"

def(main):
    #load a summary of the training data (answers (ie predictive variable data) and 
    # impact ratings (the response variable to be predicted)
    trainingDataSummary = load_and_format_training_dataset_excel(inputFileName)

    #use function to make the 5 random forest models with all training data
    randomForestModelsDict = make_five_random_forest_models(trainingDataSummary)

    #get a list of all taxa
    trainingSpeciesList = list(trainingDataSummary.index.values)

    #extract a table of only the data for answers (ie converting it into test data)
    testDataAll = get_test_data_from_training_dataset(trainingDataSummary)

    #make the predictons, output is a data frame with a column for the five predictions.
    predictionsAll = make_predictions_from_test_table(testDataAll, randomForestModelsDict)

    #transfer those predictons to a summary table
    predictionsAll = predictionsAll.add_prefix('RDTF_')
    dataSummary = trainingDataSummary.join(predictionsAll)

    #run k=n fold validation. will retrun the predictions
    predictionsKEqualsN,meansKEqualsN = k_equals_n_validation(trainingDataSummary)

    #add that to the data summary data frame
    predictionsKEqualsN = predictionsKEqualsN[['imp2','imp3','imp4','imp5','imp6']].add_prefix('RDTF_kn_')
    dataSummary = dataSummary.join(predictionsKEqualsN)
    meansKEqualsN = meansKEqualsN[['imp2','imp3','imp4','imp5','imp6']].add_prefix('means_kn_')
    dataSummary = dataSummary.join(meansKEqualsN)
    dataSummary['RDTF_imp1'] = 1.0

    trainingDataMeanImpacts = get_means_from_training_dataset(trainingDataSummary)
    trainingDataMeanImpacts.loc['imp1'] = 1.0
    trainingDataMeanImpacts = trainingDataMeanImpacts.sort_index()

    #extract the questions from the input datasheet
    questionsDict = load_questions_dict(inputFileName)

    #run imporance analyses
    questionImportanceDataFrame = get_importance_of_questions_all_taxa(trainingDataSummary,randomForestModelsDict,questionsDict)
    questionImportanceDataFrame = questionValueDataFrame


    #get the means to use as default answers
    meansDict = get_mean_values_by_question(trainingDataSummary)

    #make prediction tool worksheet
    generate_working_sheet_openpyxl(questionsDict,trainingDataSummary,randomForestModelsDict,meansDict,outputFileName)


    # import joblib
    joblib.dump({'trainingDataSummary': trainingDataSummary,\
    'randomForestModelsDict': randomForestModelsDict,\
    'trainingSpeciesList': trainingSpeciesList,\
    'testDataAll': testDataAll,\
    'predictionsAll': predictionsAll,\
    'dataSummary': dataSummary,\
    'predictionsKEqualsN': predictionsKEqualsN,\
    'meansKEqualsN': meansKEqualsN,\
    'questionImportanceDataFrame': questionImportanceDataFrame,\
    'trainingDataMeanImpacts': trainingDataMeanImpacts,\
    'questionsDict': questionsDict}, "./backup1.joblib", compress=3)

    #Do the incremental validation
    #note this is slow, and will take hours
    #note - not yet tested in master script
    incValAll,incValSummary = incremental_validation(trainingDataSummary)

    # import joblib
    joblib.dump({'trainingDataSummary': trainingDataSummary,\
    'randomForestModelsDict': randomForestModelsDict,\
    'trainingSpeciesList': trainingSpeciesList,\
    'testDataAll': testDataAll,\
    'predictionsAll': predictionsAll,\
    'dataSummary': dataSummary,\
    'predictionsKEqualsN': predictionsKEqualsN,\
    'meansKEqualsN': meansKEqualsN,\
    'incValAll': incValAll,\
    'incValSummary': incValSummary,\
    'questionImportanceDataFrame': questionImportanceDataFrame,\
    'trainingDataMeanImpacts': trainingDataMeanImpacts,\
    'questionsDict': questionsDict}, "./outputsSaved_testing.joblib", compress=3)

    import joblib

    everythingLoaded = joblib.load("./outputsSaved_testing.joblib")

    trainingDataSummary = everythingLoaded['trainingDataSummary']
    randomForestModelsDict = everythingLoaded['randomForestModelsDict']
    trainingSpeciesList = everythingLoaded['trainingSpeciesList']
    testDataAll = everythingLoaded['testDataAll']
    predictionsAll = everythingLoaded['predictionsAll']
    dataSummary = everythingLoaded['dataSummary']
    predictionsKEqualsN = everythingLoaded['predictionsKEqualsN']
    meansKEqualsN = everythingLoaded['meansKEqualsN']
    incValAll = everythingLoaded['incValAll']
    incValSummary = everythingLoaded['incValSummary']
    questionImportanceDataFrame = everythingLoaded['questionImportanceDataFrame']
    trainingDataMeanImpacts = everythingLoaded['trainingDataMeanImpacts']
    questionsDict = everythingLoaded['questionsDict']


if __name__ == "__main__":
    main()
