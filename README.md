# obj_detect
A little personal project of mine.

Basically, i want to create an object detection model
That will take an image with a square on it as an input
and will output:
-x of the upper left corner
-y of the upper left corner
-width of the square
-height of the square

In that exact order

Current best model:
model_1 with test loss = 0.07 and test accuracy = 96%

model_1 only detects white squares on a black background,
but i am now working on a model that will work with coloured images
