for file in *.md; do
    echo "Processing $file"
    python3 converter.py "$file"
done
