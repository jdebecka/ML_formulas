import math


# Error function
# Maximizing the Margin
# Error = Classification error + margin error
# Error = |w| ** 2
# Margin = 2 / |w|
# Large margin - small error
# Small margin - large error
def margin_error(calculated_error):
    return 2 / math.sqrt(calculated_error)


def classification_error(W):
    return math.fabs(sum(map(lambda w: pow(w, 2), W)))

# Minimize errors defined above using gradient descent
# C parameter - can decrease or increase margin based on the need of precision
# Error = C * classification_error + margin_error
# Large C focuses on classifying points

# # # # Polynomial kernel


