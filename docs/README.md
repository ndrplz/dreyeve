### dr(eye)ve supplementary material

a collection of resources useful for the comprehension of the paper.

#### How was Fig.7 made?
Sec 3.1 of the paper investigates on what human fixations focus on 
while driving, in terms of semantic classes. As explained in the paper,
this analysis has been carried out by normalizing the fixation map
so the maximum value equals one and then thresholding it at nine
different values ([0.00, 0.12, 0.25, 0.37, 0.50, 0.62, 0.75, 0.87, 1.00])
obtaining nine binary maps. As the threshold increases, the corresponding
binary map tightens around the fixation point, excluding from the 
analysis pixels far from them.
The following animation shows the original image (top-left), the segmented
image (top-right), binary maps at increasing thresholds (bottom-left) and 
pixel labels accounted for the construction of the histogram of each threshold.

![fig7gif](https://media.giphy.com/media/l378jbMUMxdQ4Hlza/giphy.gif "fig7gif")
 
Then, from accumulated class occurrences for each threshold we build a 
histogram, that is represented by Fig.7.

![fig7](img/fig7.png "fig7")


In this figure, by isolating the leftmost bar for each class, we obtain 
the histogram relative to the first threshold. By isolating the second one from 
left, we obtain the histogram relative to the second threshold.
Objects exibiting upward trends (e.g. road, vehicles, pedestrians) indicate that
when assigned to a non-zero probability of fixation, they are likely to be
the actual focus of the fixation point, whereas a downward trend indicates
an awareness of the object which is only circumstantial.
