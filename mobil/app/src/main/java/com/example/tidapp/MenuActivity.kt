package com.example.tidapp

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import com.google.firebase.auth.FirebaseAuth

class MenuActivity : AppCompatActivity() {

    private lateinit var btnStartCamera: Button
    private lateinit var btnViewRecords: Button
    private lateinit var btnLogout: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_menu)

        // XML tarafındaki butonları koda bağlıyoruz reisim
        btnStartCamera = findViewById(R.id.btnStartCamera)
        btnViewRecords = findViewById(R.id.btnViewRecords)
        btnLogout      = findViewById(R.id.btnLogout)

        // 1. KAMERAYI KULLAN: Bizim o tıkır tıkır çalışan ana sayfaya (MainActivity) yönlendirir
        btnStartCamera.setOnClickListener {
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        }

        // 2. GÜNCELLEME - METİNLERİ GÖRÜNTÜLE: Artık buluttaki verileri çekeceğimiz yeni sayfaya uçuruyor reisim!
        btnViewRecords.setOnClickListener {
            val intent = Intent(this, RecordsActivity::class.java)
            startActivity(intent)
        }

        // 3. ÇIKIŞ YAP: Kullanıcıyı arkadaki tüm sayfaları temizleyerek LoginActivity'ye geri atar
        btnLogout.setOnClickListener {
            // Firebase oturumunu cihazdan resmen sonlandırıyoruz reisim!
            FirebaseAuth.getInstance().signOut()

            val intent = Intent(this, LoginActivity::class.java)
            // Güvenli çıkış için geçmişteki tüm sayfaları sıfırlar
            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            startActivity(intent)
            finish()
        }
    }
}