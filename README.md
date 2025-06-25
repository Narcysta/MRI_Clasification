Opis problemu
Celem projektu jest opracowanie jak najdokładniejszego modelu klasyfikującego obrazy rezonansu magnetycznego (MRI) mózgu w celu automatycznego rozróżniania między różnymi typami nowotworów mózgu oraz przypadkiem zdrowym. 
W projekcie rozróżniamy cztery klasy obrazów MRI:
1.	Glejak mózgu (glioma) – klasa 0, agresywny nowotwór wywodzący się z komórek glejowych, często prowadzący do poważnych zaburzeń neurologicznych.
2.	Oponiak mózgu (meningioma) – klasa 1, zwykle łagodny nowotwór rozwijający się w oponach mózgowo-rdzeniowych.
3.	Rak przysadki mózgowej (pituitary tumor) – klasa 2, guz umiejscowiony w przysadce mózgowej, mogący wpływać na gospodarkę hormonalną organizmu.
4.	Zdrowy mózg – klasa 3, obrazy bez oznak nowotworu, stanowiące grupę kontrolną.
Klasyfikacja została zrealizowana przy użyciu konwolucyjnej sieci neuronowej (CNN), która uczy się rozpoznawać wzorce i cechy charakterystyczne dla każdego typu obrazu. Model został przeszkolony na zbiorze obrazów MRI, a następnie jego skuteczność oceniono z wykorzystaniem metryk dokładność (accuracy) i AUC (Area Under the Curve). Dodatkowo, cechy wyodrębnione przez sieć zostały użyte jako dane wejściowe do zewnętrznego klasyfikatora. Na podstawie tych metryk zostanie wybrany lepszy model do rozpoznawania zdjęć MRI.

Opis zastosowanego rozwiązania
1. Wstępne przetwarzanie danych
Obrazy zostały przeskalowane do rozmiaru 64x64 piksele i poddane transformacjom:
•	Konwersja do tensora – przekształcenie obrazu do formatu odpowiedniego dla PyTorch.
•	Normalizacja – standaryzacja wartości pikseli do zakresu [-1, 1], co ułatwia uczenie.

 
2. Architektura sieci CNN
Model składa się z kilku warstw:
•	3 Warstwy konwolucyjne – wykrywają cechy (linie, krawędzie)
•	Max Pooling – wybiera z tensora maksymalną wartość i przekazuje ją dalej
•	Flatten – przekształca dane z poolingu do formy 1D; wektora cech.
•	Dropout – zapobiega przeuczeniu, losowo wyłączając 20% neuronów podczas uczenia.
•	Warstwa liniowa – jest używana do klasyfikacji tensorów
  
3. Proces uczenia
Uczenie odbywało się z użyciem:
•	Funkcji kosztu CrossEntropyLoss, odpowiedniej dla klasyfikacji wieloklasowej.
•	Optymalizatora Adam, który aktualizuje wagi sieci.
•	Techniki early stopping, która zatrzymuje trening, gdy wynik walidacji przestaje się poprawiać kilka razy pod rząd.
 
 
4. Ekstrakcja cech i klasyfikator klasyczny
Sieć została potem użyta do samego wydobywania cech. Na tych cechach wytrenowano klasyczny model regresji logistycznej, który posłużył jako alternatywny klasyfikator. 
 
5. Ocena jakości modelu
Ostateczna skuteczność obu podejść (CNN i klasyfikatora klasycznego) została oceniona na zbiorze testowym przy użyciu metryk:
•	Accuracy (dokładność) – procent poprawnie sklasyfikowanych przykładów.
•	AUC (Area Under the Curve) – miara jakości klasyfikatora przy różnych progach decyzyjnych.
  

Metryki modelu
W celu oceny jakości klasyfikatora wykorzystano dwie metryki: ACC czyli dokładność oraz AUC (Area Under the Curve). Dokładność mierzy procent poprawnych klasyfikacji, natomiast AUC określa zdolność modelu do rozróżniania między poszczególnymi klasami.
Dla modelu konwolucyjnej sieci neuronowej (CNN) uzyskano następujące wyniki:
•	Accuracy: 90,08%
•	AUC: 0,9835
Dla regresji logistycznej uzyskano:
•	Accuracy: 89,70%
•	AUC: 0,9730
Wyniki te świadczą o bardzo dobrej jakości klasyfikacji (dokładność powyżej 90% dla CNN i prawie 90% dla regresji)– zarówno sieć CNN, jak i klasyfikator zewnętrzny skutecznie odróżniają 3 typy raka mózgu i zdrowego pacjenta.
Wykresy
1.Krzywa ROC
Wykres ROC pokazuje zależność między współczynnikiem True Positive Rate (TPR) a False Positive Rate (FPR) dla różnych progów decyzyjnych. Jest generowany osobno dla każdej klasy w podejściu wieloklasowym. Przerywana linia oznacza model, który losowo generuje wynik. Jeżeli krzywa modelu jest powyżej linii dla losowego generowania wyniku oznacza, że jest on lepszy od niej.
  
Wykres 1: Krzywa ROC dla modelu CNN
Krzywe dla wszystkich klas są bardzo blisko lewego górnego rogu wykresu, co oznacza, że model bardzo dobrze rozróżnia między klasami. Wartości AUC dla poszczególnych klas są wysokie (od 0.96 do 1.00), co wskazuje na bardzo dobrą zdolność modelu do klasyfikacji, z prawie idealnym rozróżnieniem. Linia przerywana to linia losowego zgadywania (AUC=0.5), a wszystkie krzywe są znacznie powyżej niej, co świadczy o wysokiej jakości modelu.


 
Wykres 2: Krzywa ROC dla klasyfikatora
Tutaj również krzywe ROC dla wszystkich klas są dobrze rozwinięte i zbliżone do lewego górnego rogu. AUC jest nieco niższe niż w przypadku CNN, ale nadal bardzo wysokie (od 0.95 do 1.00), co wskazuje, że regresja logistyczna także dobrze klasyfikuje obrazy, choć nieco słabiej niż CNN. Krzywa pomarańczowa (klasa 1) jest najniższa, co może sugerować, że ta klasa jest trudniejsza do rozróżnienia dla tego modelu. 

2. Porównanie metryk
 
Wykres 3: Porównanie ACC i AUC modelu CNN i Logistic regression
Oba modele osiągnęły bardzo dobre wyniki, z lekką przewagą CNN w obu metrykach.
3. Macierz pomyłek
Macierz pomyłek to tabela używana do oceny jakości działania modelu klasyfikacyjnego. Pokazuje, jak często model poprawnie klasyfikuje próbki i gdzie popełnia błędy.
 
Wykres 4: Macierz konfuzji dla CNN
Na macierzy widać, że większość przypadków została poprawnie przewidziana. Model miał największy problem z klasą 1, czyli oponiakiem mózgu. Mylił go ze wszystkimi klasami, głownie z glejakiem.
 
Wykres 5: Macierz konfuzji dla Regresji Logistycznej
Macierz pomyłek Regresji Logistycznej wygląda bardzo podobnie. Jak w poprzedniej macierzy model w zdecydowanej większości rozpoznał poprawnie zdjęcia. Model mylił najczęściej oponiaka z innymi przypadkami oraz mylił glejaka z oponiakiem. W skali wszystkich zdjęć są to liczby znikome.

Podsumowanie
Celem projektu było stworzenie dwóch modeli rozpoznających choroby na podstawie obrazów rezonansu magnetycznego (MR) oraz porównanie ich skuteczności. Do oceny wykorzystano metryki AUC (Area Under the Curve) oraz ACC (Accuracy). Oba modele osiągnęły bardzo wysoką dokładność, a różnice między nimi okazały się minimalne, co utrudnia jednoznaczny wybór lepszego rozwiązania.
W analizie zastosowano również macierz pomyłek, która pozwoliła ocenić, jakiego rodzaju błędy popełnia każdy z modeli – np. czy częściej nie wykrywają choroby (false negatives), czy błędnie ją diagnozują (false positives). 
Wysoki wynik AUC potwierdza, że modele skutecznie rozróżniają przypadki zdrowe od chorych, niezależnie od progu decyzyjnego. Oba modele mogą z powodzeniem mogłyby wspierać proces diagnostyczny w wykrywaniu zmian nowotworowych.
