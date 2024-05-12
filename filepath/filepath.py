# file = open("filepath/4X11999normal.txt","r",encoding='utf-8')
# normal_11999_4x = file.read().split('\n')
# file = open("filepath/4X11999canceer.txt","r",encoding='utf-8')
# canceer_11999_4x = file.read().split('\n')
# file = open("filepath/4X11693normal.txt","r",encoding='utf-8')
# normal_11693_4x = file.read().split('\n')
# file = open("filepath/4X11693canceer.txt","r",encoding='utf-8')
# canceer_11693_4x = file.read().split('\n')

file = open("filepath/10X11999canceer.txt","r",encoding='utf-8')
canceer_11999_10x = file.read().split('\n')
file = open("filepath/10X11999far canceer.txt","r",encoding='utf-8')
farcanceer_11999_10x = file.read().split('\n')
# file = open("filepath/10X11999inner canceer.txt","r",encoding='utf-8')
# innercanceer_11999_10x = file.read().split('\n')
file = open("filepath/10X11999partial.txt","r",encoding='utf-8')
partial_11999_10x = file.read().split('\n')
file = open("filepath/10X11693canceer.txt","r",encoding='utf-8')
canceer_11693_10x = file.read().split('\n')
file = open("filepath/10X11693far canceer.txt","r",encoding='utf-8')
farcanceer_11693_10x = file.read().split('\n')
# file = open("filepath/10X11693inner canceer.txt","r",encoding='utf-8')
# innercanceer_11693_10x = file.read().split('\n')
file = open("filepath/10X11693partial.txt","r",encoding='utf-8')
partial_11693_10x = file.read().split('\n')

# file_path_11999_4x = normal_11999_4x + canceer_11999_4x
# file_path_11693_4x = normal_11693_4x + canceer_11693_4x
# file_path_11999_10x = partial_11999_10x + innercanceer_11999_10x + farcanceer_11999_10x + canceer_11999_10x
# file_path_11693_10x = partial_11693_10x + innercanceer_11693_10x + farcanceer_11693_10x + canceer_11693_10x
file_path_11999_10x = partial_11999_10x + farcanceer_11999_10x + canceer_11999_10x
file_path_11693_10x = partial_11693_10x + farcanceer_11693_10x + canceer_11693_10x