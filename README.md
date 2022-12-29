# HNE-HER2-classification
CNN classification model for HNE Her2

<table>
<th>source</th>
<th>Description</th>

<tr>
<td>trainHer2ClassificationGAPAdaptive.py</td>
<td>Program to train a CNN model for IHC Her2 classification. With two conv layers, with a maxpool inbetween and GAP and FC layers at the end. Classifies to Her2 positive, Her2 negative, Her2 intermediate and other classes. This model is independent of input image size.</td>
</tr>

<tr>
<td>trainHer2ClassificationLinear.py</td>
<td>Program to train a CNN model for IHC Her2 classification. With two conv layers, and a maxpool in between and FC layers at the end. Classifies to Her2 positive, Her2 negative, Her2 intermediate and other classes.</td>
</tr>


<tr>
<td>inferHer2Classification.py</td>
<td>Program to load a trained model for Her2 classification, and label images kept on a folder to classes Her2 positive, Her2 negative, Her2 intermediate and other. Creates folders for these four classes, and keeps images inside those directories as per their predictions.</td>
</tr>

</table>
