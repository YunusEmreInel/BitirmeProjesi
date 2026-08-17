package com.example.tidapp

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.firebase.auth.FirebaseAuth

class RegisterActivity : AppCompatActivity() {

    // Firebase Auth nesnesini tanımlıyoruz
    private lateinit var auth: FirebaseAuth

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_register)

        // Firebase'i başlatıyoruz
        auth = FirebaseAuth.getInstance()

        // XML tarafındaki bileşenleri ID'leri ile bağlıyoruz (Kullanıcı adı dahil)
        val usernameField = findViewById<EditText>(R.id.etRegisterUsername) // Yeni eklenen kullanıcı adı alanı
        val emailField = findViewById<EditText>(R.id.etRegisterEmail)
        val passwordField = findViewById<EditText>(R.id.etRegisterPassword)
        val registerButton = findViewById<Button>(R.id.btnRegisterSubmit)

        // Kenarlık (System Bars) görünüm ayarı
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        // Kayıt Ol Butonuna Tıklama Olayı
        registerButton.setOnClickListener {
            val username = usernameField.text.toString().trim() // Kullanıcı adı değerini alıyoruz
            val email = emailField.text.toString().trim()
            val password = passwordField.text.toString().trim()

            // 1. Kontrol: Tüm alanlar doldurulmuş mu? (Kullanıcı adı, e-posta ve şifre)
            if (username.isNotEmpty() && email.isNotEmpty() && password.isNotEmpty()) {

                // 2. Kontrol: Şifre Firebase standartlarına uygun mu (En az 6 karakter)?
                if (password.length < 6) {
                    Toast.makeText(this, "Şifre en az 6 karakter olmalıdır!", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener // Kodun aşağıya devam etmesini engeller
                }

                // 3. Durum: Her şey doğruysa Firebase'e kayıt açıyoruz
                auth.createUserWithEmailAndPassword(email, password)
                    .addOnCompleteListener(this) { task ->
                        if (task.isSuccessful) {
                            // Kayıt başarılı olduğunda kullanıcı adı ile selamlama yapıyoruz
                            Toast.makeText(this, "Kayıt Başarılı! Hoş geldin $username", Toast.LENGTH_LONG).show()

                            // Başarılı kayıttan sonra kullanıcıyı otomatik olarak Login ekranına geri gönderiyoruz
                            val intent = Intent(this, LoginActivity::class.java)
                            startActivity(intent)
                            finish() // Geri tuşuna basınca bu ekrana tekrar dönmesin
                        } else {
                            // Kayıt başarısız (Örn: Bu maille zaten hesap açılmışsa)
                            Toast.makeText(this, "Kayıt Hatası: ${task.exception?.message}", Toast.LENGTH_LONG).show()
                        }
                    }
            } else {
                // Alanlardan herhangi biri boşsa tetiklenecek uyarı
                Toast.makeText(this, "Lütfen tüm alanları doldurun", Toast.LENGTH_SHORT).show()
            }
        }
    }
}