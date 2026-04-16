from src.claim_extraction.extractor import ClaimExtractor

extractor = ClaimExtractor()

question = "Who won the 2018 FIFA World Cup?"
answer = (
    "France won the 2018 FIFA World Cup, defeating Croatia 4–2 "
    "in the final held in Russia."
)

claims = extractor.extract(question, answer)

print("Extracted Claims:")
for c in claims:
    print("-", c)
