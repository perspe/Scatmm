#/bin/sh

# Remove previously generated file
echo "Clearing previous Versions"
rm *.html

# Build all files from markdown files
for file in *.md;
do
    echo "Converting $file...."
    pandoc -s "$file" -c "custom_style.css" -o "${file%.*}.html"
done

# Update index to conect to html
echo "Updating help.html"
sed -i "s/.md/.html/g" help.html
