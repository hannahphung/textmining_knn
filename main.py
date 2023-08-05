from textmining import TextMining

if __name__ == "__main__":
    model = TextMining()
    actual_labels = [0]*8 + [1]*8 + [2]*8
    model.visualize(actual_labels)
    model.evaluate(actual_labels)