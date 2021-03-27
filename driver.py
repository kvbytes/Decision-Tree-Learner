"""
Decision Tree Learner
CS 550 - Artificial Intelligence
Fall 2020
"""
import time

from ml_lib.ml_util import DataSet, parse_csv

from decision_tree import DecisionTreeLearner

from ml_lib.crossval import cross_validation

from statistics import mean, stdev


def main():
    """
    Machine learning with decision trees.
    Runs cross validation on data sets and reports results/trees
    """

    with open('classifier.txt', 'w') as f:
        # mushroom data, with first column being the target
        mushroomData = DataSet(name="mushrooms", target=0)

        # zoo data, with last column being the target (-1)
        # exclude animal names as it will cause one item per branch
        zooData = DataSet(name="zoo", exclude=[0])

        # decision trees for the mushroom dataset
        MushroomTree = DecisionTreeLearner(mushroomData)
        f.write("\n--------------- Mushroom Decision Tree ---------------\n")
        f.write(MushroomTree.__str__())

        pMushroomTree = DecisionTreeLearner(mushroomData, p_value=0.05)
        pMushroomTree.chi_annotate(0.05)
        pMushroomTree.prune(0.05)
        f.write("\n\n--------------- Pruned Mushroom Decision Tree ---------------\n")
        f.write(pMushroomTree.__str__())

        # decision trees for the zoo dataset
        ZooTree = DecisionTreeLearner(zooData)
        f.write("\n\n--------------- Zoo Decision Tree ---------------\n")
        f.write(ZooTree.__str__())

        pZooTree = DecisionTreeLearner(zooData, p_value=0.05)
        pZooTree.chi_annotate(0.05)
        pZooTree.prune(0.05)
        f.write("\n\n--------------- Pruned Zoo Decision Tree ---------------\n")
        f.write(pZooTree.__str__())

        resultsMushroom = cross_validation(DecisionTreeLearner, mushroomData, p_value=0.05)
        resultsZoo = cross_validation(DecisionTreeLearner, zooData, p_value=0.05)

        # mean and stdev
        f.write("\n\nMean Error for Mushroom Data = ")
        f.write(mean(resultsMushroom[0]).__str__())
        f.write("\nStandard Dev for Mushroom Data = ")
        f.write(stdev(resultsMushroom[0]).__str__())

        f.write("\n\nMean Error for Zoo Data = ")
        f.write(mean(resultsZoo[0]).__str__())
        f.write("\nStandard Dev for Zoo Data = ")
        f.write(stdev(resultsZoo[0]).__str__())


if __name__ == '__main__':
    main()
