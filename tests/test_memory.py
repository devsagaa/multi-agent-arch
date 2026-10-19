from memory import MemoryStore


def test_memory_survives_reconnection(tmp_path):
    db = str(tmp_path / "mem.db")
    s1 = MemoryStore(db)
    s1.save("I prefer bullet points", category="preference")
    s1.save("AWS Lambda has a 15 minute max timeout", category="fact")
    s1.close()

    s2 = MemoryStore(db)  # simulates an application restart
    contents = [m["content"] for m in s2.all()]
    s2.close()
    assert contents == ["I prefer bullet points", "AWS Lambda has a 15 minute max timeout"]


def test_duplicates_are_ignored(tmp_path):
    s = MemoryStore(str(tmp_path / "mem.db"))
    assert s.save("same fact") is True
    assert s.save("same fact") is False
    assert len(s.all()) == 1


def test_relevant_returns_preferences_and_matching_facts(tmp_path):
    s = MemoryStore(str(tmp_path / "mem.db"))
    s.save("I prefer concise answers", category="preference")
    s.save("AWS Lambda has a 15 minute max timeout")
    s.save("Bananas are rich in potassium")
    got = s.relevant("Is AWS Lambda good for long running agents?")
    assert got[0] == "User preference: I prefer concise answers"
    assert "Saved fact: AWS Lambda has a 15 minute max timeout" in got
    assert not any("Bananas" in g for g in got)


def test_clear(tmp_path):
    s = MemoryStore(str(tmp_path / "mem.db"))
    s.save("x")
    s.clear()
    assert s.all() == []


def test_upsert_fact_keeps_one_row_per_question(tmp_path):
    s = MemoryStore(str(tmp_path / "mem.db"))
    s.save("I prefer bullet points", category="preference")
    assert s.upsert_fact("Should I learn AWS?", "Previously researched 'Should I learn AWS?': v1", "agent:x") == "added"
    # Same question with different case/spacing/punctuation -> same row, replaced.
    assert s.upsert_fact("should i  learn AWS", "Previously researched 'should i learn AWS': v2", "agent:y") == "updated"
    facts = [m for m in s.all() if m["category"] == "fact"]
    assert len(facts) == 1 and facts[0]["content"].endswith("v2") and facts[0]["source"] == "agent:y"
    assert s.upsert_fact("Is Lambda good for agents?", "Previously researched 'x': other", "agent:x") == "added"
    assert len(s.all()) == 3  # preference + 2 distinct questions
    s.close()


def test_legacy_duplicates_are_collapsed_on_open(tmp_path):
    import sqlite3

    db = str(tmp_path / "old.db")
    # Simulate a database created before the `topic` column existed.
    c = sqlite3.connect(db)
    c.execute("""CREATE TABLE facts (id INTEGER PRIMARY KEY AUTOINCREMENT, category TEXT NOT NULL,
                 content TEXT NOT NULL UNIQUE, source TEXT NOT NULL, created_at TEXT NOT NULL)""")
    rows = [("preference", "I prefer bullet points"),
            ("fact", "Previously researched 'Should I learn AWS?': old answer"),
            ("fact", "Previously researched 'Should I learn AWS?': newer answer"),
            ("fact", "Previously researched 'Other question': keep me")]
    c.executemany("INSERT INTO facts (category, content, source, created_at) VALUES (?, ?, 'x', 't')", rows)
    c.commit(); c.close()

    s = MemoryStore(db)
    contents = [m["content"] for m in s.all()]
    assert contents == ["I prefer bullet points",
                        "Previously researched 'Should I learn AWS?': newer answer",
                        "Previously researched 'Other question': keep me"]
    # And new saves for that question now update the surviving row.
    assert s.upsert_fact("should i learn aws", "Previously researched 'q': newest", "agent:x") == "updated"
    assert len(s.all()) == 3
    s.close()
