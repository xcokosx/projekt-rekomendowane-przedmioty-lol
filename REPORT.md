# Jak działa projekt:

1. Stworzenie danych dla sieci neuronowej:
   Przy użyciu oficjalnego api od developera gry (https://developer.riotgames.com/)
   pobraliśmy dane 250 graczy o najwyższej randze w grze. Na podstawie ich "puuid" pobraliśmy
   po 50 ich ostacnich meczu. Następnie usuneliśmy duplikaty, żeby za chwilę stworzone dane nie były kilkukrotnie zduplikowane. Dzięki zdobytym "match_id" pobieramy dane o meczu i jego uczestnikach

    !!! W przypadku chęci przetestowania wymagane jest: 1. Postawienie serwera wraz ze stworzeniem tabeli z folderu data oraz podpięcie go pod riot_scraper.py . 2. Stworzenie konta na stronie developera w celu uzyskania własnego klucza API, ponieważ
    klucze te są ważne tylko 24 godziny. 3. Wymagane jest pobranie listy graczy ze strony developera oraz zapisanie jej do pliku player_data.json .
