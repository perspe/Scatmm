#/bin/sh

# Remove previously generated file
echo "Clearing previous Versions"
rm *.html

echo "Generating Header File"
pandoc -s -c "custom_style.css" "header.md" -o "header.html"
# Build all files from markdown files
for file in *.md;
do
    [ "$file" = "header.md" ] && continue
    echo "Converting $file...."
    pandoc --toc -B "header.html" -s -c "custom_style.css" "$file" -o "${file%.*}.html"
done

# Update index to conect to html
echo "Updating help.html"
sed -i "s/.md/.html/g" *.html
