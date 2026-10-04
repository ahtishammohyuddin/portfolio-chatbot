"""Quick check that retrieval finds the right section. Run: python test_retrieval.py"""
from rag import load_chunks, Retriever

chunks = load_chunks("data")
print(f"{len(chunks)} chunks loaded\n")
r = Retriever(chunks)

# (question, a word that SHOULD appear in the top result's heading)
tests = [
    ("What databases does he know?", "databases"),
    ("Tell me about the MERIDIA ERP", "MERIDIA"),
    ("Did he build anything with RFID?", "School Bus"),
    ("Where did he study?", "Education"),
    ("What does he do at DevVibe?", "DevVibe"),
    ("Does he speak Urdu?", "Languages"),
    ("What is his CGPA?", None),          # not in the data -> should return nothing
    ("What is his favourite football team?", None),
]

for q, expect in tests:
    res = r.search(q)
    top = res[0][0].heading if res else "(no match)"
    score = f"{res[0][1]:.2f}" if res else "-"
    ok = (expect is None and not res) or (expect and res and expect.lower() in top.lower())
    print(f"{'PASS' if ok else 'FAIL'}  {q!r}\n      top: {top}  score: {score}")
