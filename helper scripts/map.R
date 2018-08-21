library(rworldmap)
library(extrafont)
loadfonts()

pdf("../doc/figures/map.pdf", family="CM Roman", width=10.1, height=9.3)
map <- getMap(resolution="low")
doculects <- read.csv("coordinates.csv", header=T, encoding="UTF-8")
plot(map,
     xlim = c(2.1, 12.2),
     ylim = c(46, 55.3),
     asp=NA)
cols = c('#31688e', '#35b779', '#35b779', '#520066')
shapes = c(15, 16, 21, 17)
points(x = doculects$longitude, doculects$latitude, col=cols[doculects$type], pch=shapes[doculects$type], bg='#fafcf4', cex=1.7)

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

order = c(3, 2, 1, 4)
legend(-0.5, 48, legend=c("Ingvaeonic", "Dutch", "Central German", "Upper German"),
       col=cols[order], pch=shapes[order], cex=1.4)

shadowtext(doculects$longitude, doculects$latitude, doculects$doculect, pos=doculects$pos, offset=0.7, cex=1.4)
dev.off()
embed_fonts("../doc/figures/map.pdf")

