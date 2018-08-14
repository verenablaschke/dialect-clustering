library(rworldmap)
library(extrafont)
loadfonts()

pdf("../doc/figures/map.pdf", family="CM Roman", width=6.5, height=6.5)
map <- getMap(resolution="low")
doculects<-read.csv("coordinates.csv", header=T)
plot(map,
     xlim = c(3, 13),
     ylim = c(45.5, 56.5),
     asp=NA)
points(x = doculects$longitude, doculects$latitude, col = "red", pch=17, cex=1.5)

# https://stackoverflow.com/questions/25631216/r-is-there-any-way-to-put-border-shadow-or-buffer-around-text-labels-en-r-plot
# https://github.com/cran/TeachingDemos/blob/master/R/shadowtext.R
shadowtext <- function(x, y=NULL, labels, col='black', bg='white', 
                       theta= seq(0, 2*pi, length.out=50), r=0.2, ... ) {
  
  xy <- xy.coords(x,y)
  xo <- r*strwidth('A')
  yo <- r*strheight('A')
  
  for (i in theta) {
    text( xy$x + cos(i)*xo, xy$y + sin(i)*yo, labels, col=bg, ... )
  }
  text(xy$x, xy$y, labels, col=col, ... )
}


shadowtext(doculects$longitude, doculects$latitude, doculects$doculect, pos=doculects$pos, offset=0.5)
dev.off()
embed_fonts("../doc/figures/map.pdf")
