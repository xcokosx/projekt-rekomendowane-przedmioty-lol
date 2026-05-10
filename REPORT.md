# Jak działa projekt:

1. Stworzenie danych dla sieci neuronowej:
   Przy użyciu oficjalnego api od developera gry (https://developer.riotgames.com/)
   pobraliśmy dane 250 graczy o najwyższej randze w grze. Na podstawie ich "puuid" pobraliśmy
   po 50 ich ostacnich meczu. Następnie usuneliśmy duplikaty, żeby za chwilę stworzone dane nie były kilkukrotnie zduplikowane. Dzięki zdobytym "match_id" pobieramy dane o meczu i jego uczestnikach

   #!!!1. Stworzenie konta na stronie developera w celu uzyskania własnego klucza API, ponieważ klucze te są ważne tylko 24 godziny. 
    

   2. preprocessor_json.py konwertuje plik matches.json do preprocessed_match_champion_data.json w celu łatwiejszego dostępu do danych o meczu  
      rozegranym przez bohatera (jego sojuszników, przeciwników oraz zakupionych przedmiotów, a także jego aleję).

      Skrypt rozróżnia oraz grupuje mecze na podstawie match_id dołącza do rekordów, sojuszników oraz przeciwników z danego meczu.
      Zamienia nazwę bohatera na jego odpowiednik ID, podobnie robi z przedmiotami, oraz innymi bohaterami, którzy brali udział w grze, również zmienia on aleję na jej ID.
