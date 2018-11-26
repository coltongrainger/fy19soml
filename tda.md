```python
# load TDAstats
library("TDAstats")
library(dplyr)
```

```python
phonemes <- read.table("https://www.openml.org/data/get_csv/1592281/php8Mz7BG", header=TRUE, sep=",")
str(phonemes)
```

```python
phon1 <- filter(phonemes, Class == 1)
phon2 <- filter(phonemes, Class == 2)
data1 <- as.matrix(phon1[1:5])
data2 <- as.matrix(phon2[1:5])
colnames(data1) <- NULL
colnames(data2) <- NULL
```

```python
phon1.hom <- calculate_homology(head(data1, 1000), dim = 1)
plot_barcode(phon1.hom)
phon2.hom <- calculate_homology(head(data2, 1000), dim = 1)
plot_barcode(phon2.hom)
```

```python
angles <- runif(100, 0, 2 * pi)
azimuthal <- runif(100, 0, pi)
```

```python
S2 <- cbind(cos(angles)*sin(azimuthal), sin(angles)*sin(azimuthal), cos(azimuthal))
S2.hom <- calculate_homology(S2, dim = 2)
I3.hom <- calculate_homology(unif3d, dim = 2)
```

```python
# load ggplot2
library("ggplot2")

# plot barcodes with labels and identical axes
plot_barcode(I3.hom) +
  ggtitle("Persistent Homology for I3") +
  xlim(c(0, 2))

plot_barcode(S2.hom) +
  ggtitle("Persistent Homology for S2") +
  xlim(c(0, 2))
```
