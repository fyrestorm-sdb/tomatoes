Computer vision tomato grown or unripe classification

Using Vgg16 imagenet pre trained weigths

Top layers discarded replaced with two 4096 + one softmax 2 classes fully connected layers

Training from imagenet (390, hand labeled)

Validation from imagenet (18, hand labeled)


Results on 18 validation images:

recall #[1] 0.8888889

precision #[1] 0.9411765

F1 Score #[1] 0.9142857
