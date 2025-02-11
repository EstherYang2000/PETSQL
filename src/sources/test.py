import re

def extract_sql(query, llm='sensechat'):
    def clean_query(sql):
        """Helper function to clean up SQL syntax."""
        return sql.strip().replace('\n', ' ').replace('`', '').strip()

    # Expanded regex patterns for more flexible SQL extraction
    extraction_patterns = [
    # More flexible pattern for markdown headers
    r"\*\*(?:Answer|SQL Query|Final SQL Query|SQL):\*\*\s*(SELECT\s+.*?)(?:\n|;|$)",
    
    # Existing patterns remain the same
    r"```(?:SQL|sql|sqlite)?\s*(.*?)```",
    r"(?:###\s*SQL:|</think>|<think>)\s*(SELECT\s+.*?(?:;|$))",
    r"</think>\s*(?:.*?)(SELECT\s+.*?(?:;|$))",
    r":\s*(SELECT\s+.*?(?:;|$))",
    r"(SELECT\s+DISTINCT\s+.*?(?:;|$))",
    r"(SELECT\s+.*?(?:;|$))"
    ]

    # Try each pattern in order
    for pattern in extraction_patterns:
        matches = re.findall(pattern, query, re.DOTALL | re.IGNORECASE)
        if matches:
            # Return the last (most recent) match
            return clean_query(matches[-1])
    # Final fallback
    return clean_query(query)
# Example queries to test the function
query1 = """
<think> Alright, let's tackle this problem step by step. So, the question is asking for all distinct countries where singers above the age of 20 are from. I need to write an SQLite query that fetches this information efficiently. First, I'll look at the tables provided. The relevant table here seems to be the 'singer' table because it contains details about each singer, including their country and age. The columns in the singer table are Singer_ID, Name, Country, Song_Name, Song_release_year, Age, and Is_male. The goal is to find countries of singers who are older than 20 years old. Since we're only interested in distinct countries, I should use the DISTINCT keyword to avoid duplicates. I need to select the 'Country' from the singer table where the singer's age is greater than 20. So, my query will be selecting Country from singer with a WHERE clause that filters Age > 20. Also, considering efficiency, since this is a straightforward SELECT with a simple condition and using DISTINCT, it should execute quickly without needing complex joins or subqueries. There's no need to involve other tables like concert or stadium because the information required is all within the singer table. Putting it all together, the SQL query would be selecting the distinct Country from the singer table where Age is greater than 20. </think> To find all distinct countries of singers above age 20: SELECT DISTINCT Country FROM singer WHERE Age > 20
"""

query2 = """
<think> Okay, I need to find out which countries have singers who are older than 20 years old. Let's see how to approach this. First, looking at the singer table, it has columns like Singer_ID, Name, Country, Song_Name, Song_release_year, Age, and Is_male. The key here is the Age column because I need to filter singers where Age > 20. I should select the Country from the singer table where Age is greater than 20. Also, since the question asks for different countries, I should make sure each country appears only once in the results. That means using the DISTINCT keyword so there are no duplicates. So, putting it together: SELECT DISTINCT Country FROM singer WHERE Age > 20. This query will give me all unique countries where singers are above 20 years old. </think> To find the distinct countries with singers aged over 20: SELECT DISTINCT Country FROM singer WHERE Age > 20
"""

query3 = """<think> Alright, I need to solve this SQL problem. Let me read through what's given.  The task is to list the maximum weight and type for each type of pet. The tables involved are Pets, which has columns PetID, PetType, pet_age, and weight.   Looking at the data: - Pets table has PetID 2001 with type cat, weight 12.0 - PetID 2002 is dog, weight 13.4 - PetID 2003 is dog, weight 9.3  So for each PetType, I need to find the maximum weight.  First thought: Use a GROUP BY on PetType and apply MAX() to weight. That makes sense because grouping by type allows me to aggregate weights per type.  The SQL would look something like SELECT PetType, MAX(weight) FROM Pets GROUP BY PetType.  Wait, the problem says "list the maximum weight and type for each type of pet." So maybe I should order it by PetType or weight? Not sure if ordering is needed, but since the question doesn't specify, I'll stick to just grouping and selecting max values.  I don't think any joins are necessary here because all the data is in the Pets table. The other tables (Student and Has_Pet) aren't needed for this query.  So the final SQL should be SELECT PetType, MAX(weight) AS MaxWeight FROM Pets GROUP BY PetType. Adding an alias for clarity might help, but it's optional based on what the question wants. </think>  To list the maximum weight and type for each pet type, we group by `PetType` and select the maximum weight.  **Answer:** SELECT PetType, MAX(weight) AS MaxWeight FROM Pets GROUP BY PetType
"""

print(extract_sql(query1))
print(extract_sql(query2))
print(extract_sql(query3))
