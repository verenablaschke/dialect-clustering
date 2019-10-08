# This script creates a map with the doculect locations.
setwd('C:/Users/vbl/Documents/Uni/18 BA Thesis/thesis/helper scripts')
library(rworldmap)

pdf("../doc/figures/map-pres.pdf", width=10.1, height=9.3)

# Get the map.
map <- getMap(resolution="low")
plot(map,
     xlim = c(6, 8),
     ylim = c(46.3, 54.7),
     asp=NA)

# Plot the locations.
doculects <- read.csv("coordinates-pres.csv", header=T, encoding="UTF-8")
cols <- c('#31688e', '#35b779', '#35b779', '#520066') # blue, green, green, purple
shapes <- c(15, 16, 21, 17) # square, circle, ring, triangle
points(doculects$longitude,
       doculects$latitude,
       col=cols[doculects$type],
       pch=shapes[doculects$type],
       bg='#fafcf4', cex=2.3)

# Create a white halo around the text. Based on:
# stackoverflow.com/questions/25631216/
# github.com/cran/TeachingDemos/blob/master/R/shadowtext.R
maplabel <- function(x, y, labels,
                     theta=seq(0, 2*pi, length.out=50),
                     radius=0.2,
                     ... # Distance from coordinate, font size, direction, etc.
                     ){
  # Get the coordinates/offset.
  xy <- xy.coords(x,y)
  xo <- radius*strwidth('A')
  yo <- radius*strheight('A')
  
  # Print the white halo.
  for (i in theta) {
    text(xy$x + cos(i)*xo, xy$y + sin(i)*yo,
         labels,
         col='white', ...)
  }
  
  # Print the actual text.
  text(xy$x, xy$y, labels, ...)
}
maplabel(doculects$longitude, doculects$latitude, doculects$doculect,
         pos=doculects$pos, offset=0.7, cex=1.8)

# Add a legend.
order <- c(3, 2, 1, 4)
legend(0.7, 49.2,
       legend=c("Ingvaeonic", "Dutch", "Central German", "Upper German"),
       col=cols[order],
       pch=shapes[order], cex=1.8)

dev.off()
