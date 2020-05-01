import math
from collections import defaultdict
import pandas as pd
import numpy as np
from collections import Counter


def entropy(count_first_kind, count_second_kind):
    """
    Input: number of two different elements that we choose from
    Output: Entropy for the elements

    This formula uses summation. Instead multiplying to derive probability we add based on
    log(ab) = log(a) + log(b) principle
    """
    all_elements = count_first_kind + count_second_kind
    prob_first = count_first_kind / all_elements
    prob_second = count_second_kind / all_elements

    return -prob_first * math.log(prob_first, 2) - prob_second * math.log(prob_second, 2)


def multi_class_entropy(classes):
    """
    Input: Count of every class element that appears in a 'bucket'
    Output: Entropy for the elements

    This formula uses summation. Instead multiplying to derive probability we add based on
    log(ab) = log(a) + log(b) principle
    """
    all_elements = sum(classes)
    entropy_all = 0

    for one_count in classes:
        probability = one_count / all_elements
        entropy_all += -probability * math.log2(probability)

    return entropy_all


def information_gain(parent_entropy, child_entropy):
    """
    Input: Count of every class element that appears in a 'bucket'
    Output: Entropy for the elements

    change in entropy
    Entropy(Parent)−[m/m+n * Entropy(Child1)+ n/m+nEntropy(Child2)]
    """
    return parent_entropy - child_entropy


def split_data():
    """
    Input: Count of every class element that appears in a 'bucket'
    Output: Entropy for the elements

    implementation for ml-bugs (1).csv
    """

    data = pd.read_csv('/Users/juliadebecka/Documents/GitHub/ML_formulas/Decision_Trees/Resources/ml-bugs (1).csv')
    label_list = list(data)  # column names

    discriminators = data[label_list[0]].array.to_numpy()
    discriminatorsCounted, counts = np.unique(discriminators, return_counts=True)

    entropy_parent = entropy(counts[0], counts[1])
    colors = data[label_list[1]].array.unique().to_numpy()
    lengths = data[label_list[2]].array.unique().to_numpy(dtype=int)
    lengths.sort()

    gain_by_column = dict()
    list_count_colors = zip(discriminators, data[label_list[1]].array.to_numpy())
    list_count_lengths = zip(discriminators, data[label_list[2]].array.to_numpy(dtype=int))

    common_colors = Counter(list_count_colors).most_common()
    common_colors = dict(common_colors)
    common_lengths = Counter(list_count_lengths).most_common()
    common_lengths = dict(common_lengths)

    for discriminator in discriminators:
        for color in colors:
            list_items = list()
            key = (discriminator, color)
            list_items.append(common_colors.get(key))
            list_items.append(len(data) - sum(list_items))
            gain_by_column[(discriminator, color)] = (
                color, information_gain(entropy_parent, multi_class_entropy(list_items)))

        for i in range(1, len(lengths) - 1):
            for j in range(i, len(lengths)):
                list_items = list()
                key = (discriminator, lengths[i])
                if common_lengths.get(key) is not None:
                    list_items.append(common_lengths.get(key))
                list_items.append(len(data) - sum(list_items))

                gain_by_column[(discriminator, lengths[i])] = (
                    lengths[i], information_gain(entropy_parent, multi_class_entropy(list_items)))

    values = list(gain_by_column.values())
    print(values)
    print(max(values, key=lambda item: item[1]))


split_data()
