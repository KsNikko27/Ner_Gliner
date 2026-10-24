from postgres_storage import create_ner_factory

ner = create_ner_factory(
    output_dir=r"C:\Users\rfhg3\Desktop\Project\results",
    confidence_threshold=0.5,
)

result_file = ner.analyze_file(
    r"C:\Users\rfhg3\Desktop\Project\Gliner_text_2.txt"
)

print("Результат:", result_file)


