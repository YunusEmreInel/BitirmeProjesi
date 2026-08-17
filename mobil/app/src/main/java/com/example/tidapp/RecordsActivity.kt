package com.example.tidapp

// Tasarım (XML) bağlarının koda pürüzsüzce akması için R importu
import com.example.tidapp.R
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.database.DataSnapshot
import com.google.firebase.database.DatabaseError
import com.google.firebase.database.DatabaseReference
import com.google.firebase.database.FirebaseDatabase
import com.google.firebase.database.ValueEventListener

class RecordsActivity : AppCompatActivity() {

    private lateinit var tvCloudJsonOutput: TextView
    private lateinit var btnBackToMenu: Button
    private lateinit var databaseRef: DatabaseReference

    // SİLME İŞLEMİ İÇİN: Buluttaki benzersiz push key'leri (ID) hafızada tutacak liste
    private val kayitAnahtarlariListesi = ArrayList<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_records)

        tvCloudJsonOutput = findViewById(R.id.tvCloudJsonOutput)
        btnBackToMenu     = findViewById(R.id.btnBackToMenu)

        btnBackToMenu.setOnClickListener {
            finish()
        }

        val currentUserId = FirebaseAuth.getInstance().currentUser?.uid

        if (currentUserId == null) {
            tvCloudJsonOutput.text = "Hata: Oturum açmış bir kullanıcı bulunamadı!"
            Toast.makeText(this, "Lütfen tekrar giriş yapın.", Toast.LENGTH_LONG).show()
            return
        }

        // Avrupa sunucu URL'miz üzerinden kullanıcının kayıtlarına kilitleniyoruz
        databaseRef = FirebaseDatabase.getInstance("https://bitirmeprojesi-df808-default-rtdb.europe-west1.firebasedatabase.app/")
            .getReference("kaydedilen_metinler")
            .child(currentUserId)

        fetchCloudRecords()
    }

    private fun fetchCloudRecords() {
        databaseRef.addValueEventListener(object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                // Her veri güncellendiğinde listeyi sıfırlıyoruz reisim
                kayitAnahtarlariListesi.clear()

                if (!snapshot.exists()) {
                    tvCloudJsonOutput.text = "Bulutta henüz kaydedilmiş bir veriniz bulunmuyor reisim."
                    return
                }

                val stringBuilder = StringBuilder()
                var kayitSayisi = 1

                for (recordSnapshot in snapshot.children) {
                    try {
                        val verilerList = recordSnapshot.child("veriler").getValue() as? ArrayList<*>

                        if (verilerList != null && verilerList.isNotEmpty()) {
                            // SİLME İÇİN: Bu kaydın Firebase'deki benzersiz ID'sini listeye ekliyoruz
                            recordSnapshot.key?.let { kayitAnahtarlariListesi.add(it) }

                            // Sıra numarasını ekliyoruz: "1) "
                            stringBuilder.append("$kayitSayisi) ")

                            // 🌟 GÜNCELLEME: Kelimelerin arasına virgül yerine NOKTA (. ) koyarak birleştiriyoruz reisim!
                            stringBuilder.append(verilerList.joinToString(". "))

                            stringBuilder.append("   ❌ [SİLMEK İÇİN TIKLA]\n")
                            stringBuilder.append("--------------------------------------------------\n\n")

                            kayitSayisi++
                        }
                    } catch (e: Exception) {
                        // Hatalı düğüm olursa es geç
                    }
                }

                if (stringBuilder.isEmpty()) {
                    tvCloudJsonOutput.text = "Görüntülenecek geçerli bir metin bulunamadı reisim."
                } else {
                    tvCloudJsonOutput.text = stringBuilder.toString().trim()

                    // SİLME DOKUNUŞU: Kullanıcı listedeki bir metne tıklarsa silme diyaloğu açılacak
                    tvCloudJsonOutput.setOnClickListener {
                        silmeSecimDiyaloguGoster()
                    }
                }
            }

            override fun onCancelled(error: DatabaseError) {
                tvCloudJsonOutput.text = "Veriler yüklenirken bulut hatası oluştu: ${error.message}"
            }
        })
    }

    private fun silmeSecimDiyaloguGoster() {
        if (kayitAnahtarlariListesi.isEmpty()) return

        // Kullanıcıya hangi numaralı kaydı silmek istediğini soran şık bir liste hazırlıyoruz
        val secenekler = Array(kayitAnahtarlariListesi.size) { i -> "${i + 1}. Kaydı Tamamen Sil" }

        AlertDialog.Builder(this)
            .setTitle("🗑️ Buluttan Metin Silme Paneli")
            .setItems(secenekler) { _, hangiSira ->
                // Seçilen sıradaki kaydın Firebase anahtarını alıyoruz
                val silinecekKayitAnahtari = kayitAnahtarlariListesi[hangiSira]

                // Onay kutusu çıkartalım reisim
                AlertDialog.Builder(this)
                    .setTitle("Emin misiniz?")
                    .setMessage("${hangiSira + 1}. sıradaki metin buluttan kalıcı olarak silinecektir.")
                    .setPositiveButton("Evet, Sil") { _, _ ->
                        // Firebase'den siliyoruz!
                        databaseRef.child(silinecekKayitAnahtari).removeValue()
                            .addOnSuccessListener {
                                Toast.makeText(this@RecordsActivity, "🗑️ Kayıt buluttan başarıyla silindi!", Toast.LENGTH_SHORT).show()
                            }
                            .addOnFailureListener { e ->
                                Toast.makeText(this@RecordsActivity, "❌ Silme başarısız: ${e.message}", Toast.LENGTH_SHORT).show()
                            }
                    }
                    .setNegativeButton("İptal", null)
                    .show()
            }
            .setNegativeButton("Kapat", null)
            .show()
    }
}