struct pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};
struct ImgInfo
{
    int vpxs;
    int hpxs;
    unsigned char hdinfo[54];
    unsigned long int hbytes;
};
extern struct ImgInfo inf;
unsigned char **bmp_read(char *);
void bmp_write(unsigned char **, char *);