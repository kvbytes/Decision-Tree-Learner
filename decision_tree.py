"""
Decision Tree Learner
CS 550 - Artificial Intelligence
Fall 2020
"""

from collections import namedtuple

import numpy as np
import scipy.stats

from ml_lib.ml_util import argmax_random_tie, normalize, remove_all, best_index
from ml_lib.decision_tree_support import DecisionLeaf, DecisionFork


class DecisionTreeLearner:
    """DecisionTreeLearner - Class to learn decision trees and predict classes
    on novel exmaples
    """

    # Typedef for method chi2test result value (see chi2test for details)
    chi2_result = namedtuple("chi2_result", ('value', 'similar'))

    def __init__(self, dataset, debug=False, p_value=None):
        """
        DecisionTreeLearner(dataset)
        dataset is an instance of ml_lib.ml_util.DataSet.
        """

        # Hints: Be sure to read and understand the DataSet class
        # as you will use it throughout.

        # ---------------------------------------------------------------
        # Do not modify these lines, the unit tests will expect these fields
        # to be populated correctly.
        self.dataset = dataset

        # degrees of freedom for Chi^2 tests is number of categories minus 1
        self.dof = len(self.dataset.values[self.dataset.target]) - 1

        # Learn the decison tree
        self.tree = self.decision_tree_learning(dataset.examples, dataset.inputs)
        # -----------------------------------------------------------------

        self.debug = debug

    def __str__(self):
        "str - Create a string representation of the tree"
        if self.tree is None:
            result = "untrained decision tree"
        else:
            result = str(self.tree)  # string representation of tree
        return result

    def decision_tree_learning(self, examples, attrs, parent=None, parent_examples=()):
        """
        decision_tree_learning(examples, attrs, parent_examples)
        Recursively learn a decision tree
        examples - Set of examples (see DataSet for format)
        attrs - List of attribute indices that are available for decisions
        parent - When called recursively, this is the parent of any node that
           we create.
        parent_examples - When not invoked as root, these are the examples
           of the prior level.
        """
        # from pseudo code in lecture slides
        # pick whatever parent had most of
        if len(examples) == 0:
            result = DecisionLeaf(self.plurality_value(parent_examples), self.count_targets(parent_examples), parent)
            return result
        elif self.all_same_class(examples):
            result = DecisionLeaf(examples[0][self.dataset.target], self.count_targets(examples), parent)
            return result
        # no more questions to ask
        elif len(attrs) == 0:
            result = DecisionLeaf(self.plurality_value(examples), self.count_targets(examples), parent)
            return result
        else:
            a = argmax_random_tie(attrs, key=lambda attr: self.information_gain(attr, examples))
            t = DecisionFork(a, self.count_targets(examples), parent=parent)
            for val, vexamples in self.split_by(a, examples):
                subtree = self.decision_tree_learning(vexamples, remove_all(a, attrs), t, examples)
                t.add(val, subtree)
            result = t
            return result

    def plurality_value(self, examples):
        """
        Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality).
        """
        popular = argmax_random_tie(self.dataset.values[self.dataset.target],
                                    key=lambda v: self.count(self.dataset.target, v, examples))
        return popular

    def count(self, attr, val, examples):
        """Count the number of examples that have example[attr] = val."""
        return sum(e[attr] == val for e in examples)

    def count_targets(self, examples):
        """count_targets: Given a set of examples, count the number of examples
        belonging to each target.  Returns list of counts in the same order
        as the DataSet values associated with the target
        (self.dataset.values[self.dataset.target])
        """

        tidx = self.dataset.target  # index of target attribute
        target_values = self.dataset.values[tidx]  # Class labels across dataset

        # Count the examples associated with each target
        counts = [0 for i in target_values]
        for e in examples:
            target = e[tidx]
            position = target_values.index(target)
            counts[position] += 1

        return counts

    def all_same_class(self, examples):
        """Are all these examples in the same target class?"""
        class0 = examples[0][self.dataset.target]
        return all(e[self.dataset.target] == class0 for e in examples)

    def choose_attribute(self, attrs, examples):
        """Choose the attribute with the highest information gain."""
        # gains = []
        # for val, vexamples in self.split_by(attrs, examples):
        #    gains.append(self.information_gain())
        attribute = argmax_random_tie(attrs,
                                      key=lambda attr: self.information_gain(attr, examples))
        return attribute

    def information_gain(self, attr, examples):
        """Return the expected reduction in entropy for examples from splitting by attr."""
        # FORMULA: summation( (p_k+n_k)/(p+n) * B(p_k/(p_k+n_k)) )
        # (p_k+n_k)/(p+n) = count of vexamples/count of examples
        remainder = []
        for val, vexamples in self.split_by(attr, examples):
            remainder.append(self.information_content(self.count_targets(vexamples)) * (len(vexamples) / len(examples)))
        remainder = np.sum(remainder)

        entropy = self.information_content(self.count_targets(examples))
        gain = entropy - remainder
        return gain

    def split_by(self, attr, examples):
        """split_by(attr, examples)
        Return a list of (val, examples) pairs for each val of attr.
        """
        return [(v, [e for e in examples if e[attr] == v]) for v in self.dataset.values[attr]]

    def predict(self, x):
        "predict - Determine the class, returns class index"
        return self.tree(x)  # Evaluate the tree on example x

    def __repr__(self):
        return repr(self.tree)

    @classmethod
    def information_content(cls, class_counts):
        """Given an iterable of counts associated with classes
        compute the empirical entropy.

        Example:  3 class problem where we have 3 examples of class 0,
        2 examples of class 1, and 0 examples of class 2:
        information_content((3, 2, 0)) returns ~ .971
        """

        # Hints:
        #  Remember discrete values use log2 when computing probability
        #  Function normalize might be helpful here...
        #  Python treats logarithms of 0 as a domain error, whereas numpy
        #    will compute them correctly.  Be careful.

        # FORMULA: summation( P(x_i) * log2(1/P(x_i)) ) = -summation( P(x_i) * log2(P(x_i)) )
        Px = normalize(remove_all(0, class_counts))
        PxArray = []
        for xi in Px:
            PxArray.append(xi * np.log2(xi))
        entropy = -np.sum(PxArray)
        return entropy

    def information_per_class(self, examples):
        """information_per_class(examples)
        Given a set of examples, use the target attribute of the dataset
        to determine the information associated with each target class
        Returns information content per class.
        """
        # Hint:  list of classes can be obtained from
        # self.data.set.values[self.dataset.target]

        # FORMULA: I(x_i) = log2(1/P(x_i))
        Px = normalize(self.count_targets(examples))
        IxArray = []
        for xi in Px:
            IxArray.append(np.log2(1 / xi))
        return IxArray

    def prune(self, p_value):
        """Prune leaves of a tree when the hypothesis that the distribution
        in the leaves is not the same as in the parents as measured by
        a chi-squared test with a significance of the specified p-value.

        Pruning is only applied to the last DecisionFork in a tree.
        If that fork is merged (DecisionFork and child leaves (DecisionLeaf),
        the DecisionFork is replaced with a DecisionLeaf.  If a parent of
        and DecisionFork only contains DecisionLeaf children, after
        pruning, it is examined for pruning as well.
        """

        # Hint - Easiest to do with a recursive auxiliary function, that takes
        # a parent argument, but you are free to implement as you see fit.
        # e.g. self.prune_aux(p_value, self.tree, None)
        # Call recursive helper function when there is a p-value
        if p_value is not None:
            self.prune_aux(p_value, self.tree)

    def prune_aux(self, p_value, tree):
        # perform a post-order tree traversal to find last DecisionFork:
        childList = tree.branches.values()
        for treechild in childList:
            # verify that the child is a fork
            if isinstance(treechild, DecisionFork):
                # check chi2 to find if we need to prune
                checkChi2 = self.chi2test(p_value, self.prune_aux(p_value, treechild))
                if checkChi2.similar is True:
                    # iterate though all the branches to find location of the child we want to prune
                    prunechild = self.prune_aux(p_value, treechild)
                    for val in tree.branches.values():
                        treeLeafCounter = 0
                        if val is prunechild:
                            # leaf value to replace the fork
                            bestval = self.dataset.values[self.dataset.target][best_index(val.distribution)]
                            tree.branches[list(tree.branches.keys())[treeLeafCounter]] = DecisionLeaf(bestval,
                                                                                                      val.distribution,
                                                                                                      val.parent)
                        treeLeafCounter += 1
        return tree

    def chi_annotate(self, p_value):
        """chi_annotate(p_value)
        Annotate each DecisionFork with the tuple returned by chi2test
        in attribute chi2.  When present, these values will be printed along
        with the tree.  Calling this on an unpruned tree can significantly aid
        with developing pruning routines and verifying that the chi^2 statistic
        is being correctly computed.
        """
        # Call recursive helper function
        self.__chi_annotate_aux(self.tree, p_value)

    def __chi_annotate_aux(self, branch, p_value):
        """chi_annotate(branch, p_value)
        Add the chi squared value to a DecisionFork.  This is only used
        for debugging.  The decision tree helper functions will look for a
        chi2 attribute.  If there is one, they will display chi-squared
        test information when the tree is printed.
        """

        if isinstance(branch, DecisionLeaf):
            return  # base case
        else:
            # Compute chi^2 value of this branch
            branch.chi2 = self.chi2test(p_value, branch)
            # Check its children
            for child in branch.branches.values():
                self.__chi_annotate_aux(child, p_value)

    def chi2test(self, p_value, fork):
        """chi2test - Helper function for prune
        Given a DecisionFork and a p_value, determine if the children
        of the decision have significantly different distributions than
        the parent.

        Returns named tuple of type chi2result:
        chi2result.value - Chi^2 statistic
        chi2result.similar - True if the distribution in the children of the
           specified fork are similar to the the distribution before the
           question is asked.  False indicates that they are not similar and
           that there is a significant difference between the fork and its
           children
        """
        if not isinstance(fork, DecisionFork):
            raise ValueError("fork is not a DecisionFork")

        # Hint:  You need to extend the 2 case chi^2 test that we covered
        # in class to an n-case chi^2 test.  This part is straight forward.
        # Whereas in class we had positive and negative samples, now there
        # are more than two, but they are all handled similarly.

        # Don't forget, scipy has an inverse cdf for chi^2
        # scipy.stats.chi2.ppf

        # FORMULA: Expected in each split if the distribution does not change:

        # phat_k = p * ( p_k ) / (p)

        # DELTA = summation (( p_k - phat_k)^2 ) / (phat_k)
        # print("Chi2 test...................................................")

        counter = 0
        ep_k = 0
        deltaArr = []
        distributionSize = len(fork.distribution)
        childList = fork.branches.values()

        for x in childList:
            for val in range(distributionSize):
                # items in split * child distribution/expected rate
                p = x.distribution[val]
                p_k = fork.distribution[val]
                expected = sum(fork.distribution)
                ##print("expected", expected)
                childSum = sum(x.distribution)
                ##print("childsum", childSum)
                ep_k = p_k * childSum / expected
                # make an array of delta
                if ep_k > 0 or ep_k < 0:
                    deltaArr.append(((p - ep_k) ** 2) / ep_k)
                counter = counter + 1

        # Compute χ2 statistic ∆, sum all the delta values
        delta = sum(deltaArr)

        # calculating Δt
        deltaT = scipy.stats.chi2.ppf(1 - p_value, self.dof)

        # calculating Δ<Δt prune, Δ>=Δt retain
        if delta < deltaT:
            # print("prune")
            prune = True
        else:
            # print("retain")
            prune = False

        # create the named tuple with chi2 statistic and the similarity in distribution
        result = namedtuple('chi2result', ['value', 'similar'])

        return result(delta, prune)

    def __str__(self):
        """str - String representation of the tree"""
        return str(self.tree)
